import abc
import logging
import time
from typing import Generator, Optional, List

import simpy
from ether.util import parse_size_string
from faas.system import FunctionRequest, FunctionResponse, FunctionContainer, FunctionImage, Function, \
    ScalingConfiguration
from faas.util.constant import controller_role_label, hostname_label, client_role_label, zone_label, function_label, \
    api_gateway_type_label
from skippy.core.scheduler import Scheduler

from examples.decentralized_clients.main import get_resnet50_inference_cpu_image_properties, \
    get_resnet50_training_cpu_image_properties, get_galileo_worker_image_properties, \
    extract_dfs, prepare_function_deployments
from examples.decentralized_loadbalancers.topology import testbed_topology
from examples.decentralized_loadbalancers.wrr import LeastResponseTimeLoadBalancer
from examples.util.clients import find_clients
from examples.watchdogs.inference import InferenceFunctionSim
from examples.watchdogs.training import TrainingFunctionSim
from sim import docker
from sim.benchmark import Benchmark
from sim.context.platform.deployment.model import SimFunctionDeployment, DeploymentRanking, SimScalingConfiguration
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionSimulator, SimFunctionReplica, SimulatorFactory
from sim.faas.core import SimResourceConfiguration, RoundRobinLoadBalancer, \
    SimLoadBalancer, Node, LocalizedRoundRobinBalancer
from sim.faassim import Simulation
from sim.predicates import PodHostEqualsNode
from sim.requestgen import SimpleFunctionRequestFactory, expovariate_arrival_profile, \
    constant_rps_profile

logger = logging.getLogger(__name__)


class LoadBalancerFunctionContainer(FunctionContainer):
    def __init__(self, fn_container: FunctionContainer):
        super(LoadBalancerFunctionContainer, self).__init__(fn_container.fn_image, fn_container.resource_config,
                                                            fn_container.labels)


def create_scheduler(env: Environment):
    scheduler = Scheduler(env.cluster)
    scheduler.predicates.append(PodHostEqualsNode())
    # scheduler.predicates.append(CheckNodeLabelPresencePred([client_label]))
    return scheduler


class ForwardingClientFunctionContainer(FunctionContainer):
    """
    This class extends the regular FunctionContainer to include objects that are used to generate requests.
    """
    ia_generator: Generator
    size: int
    fn: SimFunctionDeployment
    lb_fn: SimFunctionDeployment
    # if True, we consider this to be the maximum number of requests that should be generated
    # if False, it is considered to be the duration the client will generate requests
    max_requests: Optional[int]

    def __init__(self, fn_container: FunctionContainer, ia_generator: Generator,
                 size: int, fn: SimFunctionDeployment,
                 lb_fn: SimFunctionDeployment,
                 max_requests: Optional[int] = None):
        super(ForwardingClientFunctionContainer, self).__init__(fn_container.fn_image, fn_container.resource_config,
                                                                fn_container.labels)
        self.ia_generator = ia_generator
        self.lb_fn = lb_fn
        self.size = size
        self.fn = fn
        self.max_requests = max_requests


class ForwardingClientSimulator(FunctionSimulator):
    """
    This FunctionSimulator simulates a client that invokes a function.
    The advantage of that is, that the simulation will simulate any network traffic accurately.
    Which entails the function call between the client and the final destination (invoked function replica).
    """

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, Optional[FunctionResponse]]:
        now = env.now
        try:
            container: ForwardingClientFunctionContainer = replica.container
            ia_generator = container.ia_generator
            max_requests = None
            if container.max_requests:
                max_requests = container.max_requests

            if max_requests is None:
                while True:
                    ia = next(ia_generator)
                    request = FunctionRequest(
                        container.lb_fn.name,
                        now,
                        client=replica.node.name,
                        size=container.size,
                        body=container.fn.name
                    )
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request)
            else:
                for _ in range(max_requests):
                    ia = next(ia_generator)
                    request = FunctionRequest(
                        container.lb_fn.name,
                        now,
                        client=replica.node.name,
                        size=container.size,
                        body=container.fn.name
                    )
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request)

        except simpy.Interrupt:
            pass
        except StopIteration:
            logger.debug(f'{replica.function.name} gen has finished')
        except Exception as e:
            logger.error(e)
        finally:
            # return FunctionResponse(request, request.request_id, request.client, request.name, request.body, 200, None,
            #                         None, None, None)
            return None


class BaseLoadBalancerSimulator(FunctionSimulator, abc.ABC):
    """
    This FunctionSimulator acts as base for implementations Load Balancers that are scheduled as Pods.
    This allows for decentralized clients and load balancers and enables a full edge-cloud simulation.
    """

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, Optional[FunctionResponse]]:
        next_replica = self.next_replica(env, replica, request)
        host = replica.node.name
        proxy_request = self._create_proxy_request(env, host, next_replica, request)
        response = yield from env.faas.invoke(proxy_request)
        # TODO might be interesting to insert here some headers (i.e., when the load balancer received the request,...)
        return response

    def next_replica(self, env: Environment, replica: SimFunctionReplica,
                     request: FunctionRequest) -> SimFunctionReplica:
        ...

    def _create_proxy_request(self, env: Environment, host: str, replica: SimFunctionReplica,
                              request: FunctionRequest) -> FunctionRequest:
        fn = request.body if replica.labels[function_label] == request.body else replica.function.name
        forwarded_true = {'lb-forwarded': True}
        if request.headers is not None:
            request.headers.update(forwarded_true)
        else:
            request.headers = forwarded_true
        return FunctionRequest(
            name=fn,
            start=env.now,
            size=request.size,
            request_id=request.request_id,
            body=request.body,
            client=host,
            replica=replica,
            headers=request.headers
        )


class LoadBalancerSimulator(BaseLoadBalancerSimulator):

    def __init__(self, lb: SimLoadBalancer):
        self.lb = lb

    def next_replica(self, env: Environment, replica: SimFunctionReplica,
                     request: FunctionRequest) -> SimFunctionReplica:
        modified_request = self.copy_request(replica.node.name, request)
        return self.lb.next_replica(modified_request)

    def copy_request(self, host: str, request: FunctionRequest) -> FunctionRequest:
        return FunctionRequest(
            request.body,
            start=request.start,
            size=request.size,
            request_id=request.request_id,
            body=request.body,
            client=host,
            headers=request.headers
        )


class DecentralizedAIFunctionSimulatorFactory(SimulatorFactory):

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        if 'inference' in fn.fn_image.image:
            return InferenceFunctionSim(4)
        elif 'training' in fn.fn_image.image:
            return TrainingFunctionSim()
        elif 'galileo-worker' in fn.fn_image.image:
            return ForwardingClientSimulator()
        elif 'rr-balancer' == fn.fn_image.image:
            return LoadBalancerSimulator(RoundRobinLoadBalancer(env))
        elif 'localized-lrt-balancer' == fn.fn_image.image:
            # TODO let user pass more parameters, possibly via labels
            cluster = fn.labels[zone_label]
            return LoadBalancerSimulator(LeastResponseTimeLoadBalancer(env, cluster))
        elif 'localized-rr-balancer' == fn.fn_image.image:
            cluster = fn.labels[zone_label]
            return LoadBalancerSimulator(LocalizedRoundRobinBalancer(env, cluster))


def create_load_balancer_deployment(lb_id: str, type: str, host: str, cluster: str):
    lb_fn_name = f'lb-' + lb_id + "-" + host
    lb_image_name = type
    lb_image = FunctionImage(image=lb_image_name)
    lb_fn = Function(lb_fn_name, fn_images=[lb_image],
                     labels={function_label: api_gateway_type_label, controller_role_label: 'true',
                             zone_label: cluster})

    fn_container = FunctionContainer(lb_image, SimResourceConfiguration(),
                                     {function_label: api_gateway_type_label, controller_role_label: 'true',
                                      hostname_label: host, zone_label: cluster})

    lb_container = LoadBalancerFunctionContainer(fn_container)

    lb_fd = SimFunctionDeployment(
        lb_fn,
        [lb_container],
        SimScalingConfiguration(ScalingConfiguration(scale_min=1, scale_max=1, scale_zero=False)),
        DeploymentRanking([lb_image_name])
    )

    return lb_fd


def get_go_load_balancer_image_props(name: str) -> List[ImageProperties]:
    return [
        # image size from go-load-balancer
        ImageProperties(name, parse_size_string('10M'), arch='arm32v7'),
        ImageProperties(name, parse_size_string('10M'), arch='x86'),
        ImageProperties(name, parse_size_string('10M'), arch='arm64v8'),
    ]


def prepare_load_balancer_deployments(type: str, hosts: List[Node]) -> List[SimFunctionDeployment]:
    def create_id(i: int):
        return f'load-balancer-{i}'

    lbs = []
    for idx, host in enumerate(hosts):
        lb_id = create_id(idx)
        lbs.append(create_load_balancer_deployment(lb_id, type, host.name, host.labels[zone_label]))

    return lbs


class DecentralizedLoadBalancerTrainInferenceBenchmark(Benchmark):
    def __init__(self, balancer: str, clients: List[Node], load_balancers_hosts: List[Node]):
        self.balancer = balancer
        self.clients = clients
        self.load_balancer_hosts = load_balancers_hosts

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        images = []
        resnet50_inference_cpu_img_properties = get_resnet50_inference_cpu_image_properties()
        resnet50_training_cpu_img_properties = get_resnet50_training_cpu_image_properties()
        galileo_worker_image_properties = get_galileo_worker_image_properties()
        lb_image_properties = get_go_load_balancer_image_props(self.balancer)

        images.extend(resnet50_inference_cpu_img_properties)
        images.extend(resnet50_training_cpu_img_properties)
        images.extend(galileo_worker_image_properties)
        images.extend(lb_image_properties)

        for image in images:
            containers.put(image)

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        deployments = []
        function_deployments = prepare_function_deployments()
        load_balancer_deployment = prepare_load_balancer_deployments(self.balancer, self.load_balancer_hosts)
        client_deployments = self.prepare_client_deployments_for_experiment(load_balancer_deployment,
                                                                            function_deployments)

        deployments.extend(function_deployments)
        deployments.extend(client_deployments)
        deployments.extend(load_balancer_deployment)

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        # block until replicas become available (scheduling has finished and replicas have been deployed on the node)
        logger.info('waiting for replicas')
        for deployment in deployments:
            yield env.process(env.faas.poll_available_replica(deployment.name))

        # run workload
        ps = []
        request_factory = SimpleFunctionRequestFactory()
        for deployment in client_deployments:
            ps.append(env.process(env.faas.invoke(request_factory.generate(env, deployment))))

        # wait for invocation processes to finish
        for p in ps:
            yield p

    def prepare_client_deployments_for_experiment(self, lb_deployments: List[SimFunctionDeployment],
                                                  deployments: List[SimFunctionDeployment]) -> List[
        SimFunctionDeployment]:

        fds = []
        inference_clients = [self.clients[0].name]
        training_clients = [self.clients[1].name]
        # we assume that the file size is 250KB
        inference_size = 250
        # we assume that the file size is 10MB (100000KB)
        training_size = 10000
        fds.extend(prepare_inference_clients(inference_clients, inference_size, deployments[0], lb_deployments[0]))
        fds.extend(prepare_training_clients(training_clients, training_size, deployments[1], lb_deployments[2]))

        return fds


def prepare_client_deployment(client_id: str, host: str, ia_generator, max_requests, size: int,
                              deployment: SimFunctionDeployment, lb_deployment: SimFunctionDeployment):
    # Design time
    client_fn_name = f'client-{deployment.name}-' + client_id + "-" + host
    client_image_name = 'galileo-worker'
    client_image = FunctionImage(image=client_image_name)
    client_fn = Function(client_fn_name, fn_images=[client_image])

    fn_container = FunctionContainer(client_image, SimResourceConfiguration(),
                                     {client_role_label: 'true',
                                      hostname_label: host})
    client_container = ForwardingClientFunctionContainer(
        fn_container,
        ia_generator,
        size,
        deployment,
        lb_deployment,
        max_requests=max_requests,
    )

    client_fd = SimFunctionDeployment(
        client_fn,
        [client_container],
        SimScalingConfiguration(ScalingConfiguration(scale_min=1, scale_max=1, scale_zero=False)),
        DeploymentRanking([client_image_name])
    )
    return client_fd


def prepare_client_deployments(ia_generator, client_names: List[str], deployment: SimFunctionDeployment,
                               lb_deployment: SimFunctionDeployment, max_requests: int, size: int):
    client_fds = []
    for idx, client in enumerate(client_names):
        client_inference_fd = prepare_client_deployment(
            str(idx),
            client,
            ia_generator,
            max_requests,
            size,
            deployment,
            lb_deployment
        )
        client_fds.append(client_inference_fd)

    return client_fds


def prepare_inference_clients(clients: List[str], size: int, deployment: SimFunctionDeployment,
                              lb_deployment: SimFunctionDeployment) -> List[
    SimFunctionDeployment]:
    inference_max_rps = 5
    inference_max_requests = 200
    # generate profile
    inference_ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=inference_max_rps), max_ia=1)
    logger.info(
        f'executing resnet50-inference requests with {inference_max_rps} rps and maximum {inference_max_requests}')
    inference_client_fds = prepare_client_deployments(
        inference_ia_generator,
        clients,
        deployment,
        lb_deployment,
        inference_max_requests,
        size
    )
    return inference_client_fds


def prepare_training_clients(clients: List[str], size: int, deployment: SimFunctionDeployment,
                             lb_deployment: SimFunctionDeployment) -> List[
    SimFunctionDeployment]:
    training_max_rps = 1
    training_max_requests = 10
    logger.info(
        f'executing resnet50-training requests with {training_max_rps} rps and maximum {training_max_requests}')
    # generate profile
    training_ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=training_max_rps), max_ia=1)

    training_client_fds = prepare_client_deployments(
        training_ia_generator,
        clients,
        deployment,
        lb_deployment,
        training_max_requests,
        size
    )

    return training_client_fds


def find_lbs(topology):
    return [x for x in topology.get_nodes() if x.labels.get(controller_role_label) is not None]


def execute_benchmark():
    topology = testbed_topology()

    clients = find_clients(topology)

    lbs = find_lbs(topology)
    balancer = 'localized-rr-balancer'
    # balancer = 'rr-balancer'
    benchmark = DecentralizedLoadBalancerTrainInferenceBenchmark(balancer, clients, lbs)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(topology, benchmark)
    sim.create_simulator_factory = DecentralizedAIFunctionSimulatorFactory

    # override the SimulatorFactory factory
    sim.env.benchmark = benchmark
    sim.env.topology = topology
    sim.init_environment(sim.env)
    sim.env.scheduler = create_scheduler(sim.env)

    # run the simulation
    start = time.time()
    sim.run()
    end = time.time()
    duration = end - start

    return duration, sim


def main():
    logging.basicConfig(level=logging.DEBUG)
    logger.info('Start decentralized load balancers example.')
    duration, sim = execute_benchmark()
    env = sim.env
    dfs = extract_dfs(sim)
    logger.info(f'Time passed in simulation: {env.now}, wall time passed: {duration}')
    logger.info('Mean exec time %d', dfs['invocations_df']['t_exec'].mean())
    logger.info(f'Fets invocations: {len(dfs["fets_df"])}')

    logger.info('End decentralized load balancers example.')


if __name__ == '__main__':
    main()
