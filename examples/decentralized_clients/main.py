import logging
import time
from typing import Generator, Optional, List

import simpy
from ether.util import parse_size_string
from faas.system import FunctionRequest, FunctionResponse, FunctionContainer, FunctionImage, Function, \
    ScalingConfiguration
from faas.util.constant import client_role_label, hostname_label, worker_role_label, function_label
from skippy.core.scheduler import Scheduler

from examples.decentralized_loadbalancers.topology import testbed_topology
from examples.util.clients import find_clients
from examples.watchdogs.inference import InferenceFunctionSim
from examples.watchdogs.training import TrainingFunctionSim
from sim import docker
from sim.benchmark import Benchmark
from sim.context.platform.deployment.model import SimFunctionDeployment, SimScalingConfiguration, DeploymentRanking
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionSimulator, SimFunctionReplica, SimulatorFactory
from sim.faas.core import SimResourceConfiguration, Node
from sim.faassim import Simulation
from sim.predicates import PodHostEqualsNode
from sim.requestgen import FunctionRequestFactory, SimpleFunctionRequestFactory, expovariate_arrival_profile, \
    constant_rps_profile

logger = logging.getLogger(__name__)

"""
This example shows how to generate requests from different nodes.
Specifically, we want to include the network latency between client nodes (up to users to select)
and function hosting nodes.
This is done via scheduling Pods that simulate client behavior (i.e., invoking a function)
This behavior is implemented in the ClientSimulator, which we include in the SimulatorFactory.
This lets us create FunctionDeployments that, when invoked, will generate requests. 
Note, that while the requests are made from the clients, the load balancing decision will still happen centralized.
"""


class ClientFunctionContainer(FunctionContainer):
    """
    This class extends the regular FunctionContainer to include objects that are used to generate requests.
    """
    ia_generator: Generator
    fn_request_factory: FunctionRequestFactory
    fn: SimFunctionDeployment
    # if True, we consider this to be the maximum number of requests that should be generated
    # if False, it is considered to be the duration the client will generate requests
    max_requests: Optional[int]

    def __init__(self, fn_container: FunctionContainer, ia_generator: Generator,
                 fn_request_factory: FunctionRequestFactory, fn: SimFunctionDeployment,
                 max_requests: Optional[int] = None):
        super(ClientFunctionContainer, self).__init__(fn_container.fn_image, fn_container.resource_config,
                                                      fn_container.labels)
        self.ia_generator = ia_generator
        self.fn_request_factory = fn_request_factory
        self.fn = fn
        self.max_requests = max_requests


class ClientSimulator(FunctionSimulator):
    """
    This FunctionSimulator simulates a client that invokes a function.
    The advantage of that is, that the simulation will simulate any network traffic accurately.
    Which entails the function call between the client and the final destination (invoked function replica).
    """

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, Optional[FunctionResponse]]:
        container: ClientFunctionContainer = replica.container
        request_factory: SimpleFunctionRequestFactory = container.fn_request_factory
        request_factory.client = replica.node.name

        # read generator parameters
        fn_deployment = replica.container.fn
        try:
            container: ClientFunctionContainer = replica.container
            ia_generator = container.ia_generator
            max_requests = None
            if container.max_requests:
                max_requests = container.max_requests

            if max_requests is None:
                while True:
                    ia = next(ia_generator)
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request_factory.generate(env, fn_deployment))
            else:
                for _ in range(max_requests):
                    ia = next(ia_generator)
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request_factory.generate(env, fn_deployment))

        except simpy.Interrupt:
            pass
        except StopIteration:
            logger.debug(f'{replica.function.name} gen has finished')
        finally:
            # return FunctionResponse(request, request.request_id, request.client, request.name, request.body, 200, None,
            #                         None, None, None)
            return None


def create_scheduler(env: Environment):
    scheduler = Scheduler(env.cluster)
    scheduler.predicates.append(PodHostEqualsNode())
    # scheduler.predicates.append(CheckNodeLabelPresencePred([client_label]))
    return scheduler


class ClientAIFunctionSimulatorFactory(SimulatorFactory):

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        if 'inference' in fn.fn_image.image:
            return InferenceFunctionSim(4)
        elif 'training' in fn.fn_image.image:
            return TrainingFunctionSim()
        elif 'galileo-worker' in fn.fn_image.image:
            return ClientSimulator()


def prepare_client_deployment(client_id: str, host: str, ia_generator, max_requests, request_factory,
                              deployment: SimFunctionDeployment):
    # Design time
    client_fn_name = f'client-{deployment.name}-' + client_id + "-" + host
    client_image_name = 'galileo-worker'
    client_image = FunctionImage(image=client_image_name)
    client_fn = Function(client_fn_name, fn_images=[client_image])

    fn_container = FunctionContainer(client_image, SimResourceConfiguration(),
                                     {client_role_label: 'true',
                                      hostname_label: host})
    client_container = ClientFunctionContainer(fn_container, ia_generator, request_factory,
                                               deployment,
                                               max_requests=max_requests)

    client_fd = SimFunctionDeployment(
        client_fn,
        [client_container],
        SimScalingConfiguration(ScalingConfiguration(scale_min=1, scale_max=1, scale_zero=False)),
        DeploymentRanking([client_image_name])
    )
    return client_fd


def prepare_resnet_inference_deployment():
    # Design time

    resnet_inference = 'resnet50-inference'
    inference_cpu = 'resnet50-inference-cpu'

    resnet_inference_cpu = FunctionImage(image=inference_cpu)
    resnet_fn = Function(resnet_inference, fn_images=[resnet_inference_cpu], labels={function_label: resnet_inference})

    # Run time

    resnet_cpu_container = FunctionContainer(resnet_inference_cpu, SimResourceConfiguration(),
                                             {worker_role_label: "true", function_label: resnet_inference})

    resnet_fd = SimFunctionDeployment(
        resnet_fn,
        [resnet_cpu_container],
        SimScalingConfiguration(),
        DeploymentRanking([inference_cpu])
    )

    return resnet_fd


def prepare_resnet_training_deployment():
    # Design time

    resnet_training = 'resnet50-training'
    training_cpu = 'resnet50-training-cpu'

    resnet_training_cpu = FunctionImage(image=training_cpu)
    resnet_fn = Function(resnet_training, fn_images=[resnet_training_cpu], labels={function_label: resnet_training})

    # Run time

    resnet_cpu_container = FunctionContainer(resnet_training_cpu, SimResourceConfiguration(),
                                             labels={function_label: resnet_training, worker_role_label: 'true'})

    resnet_fd = SimFunctionDeployment(
        resnet_fn,
        [resnet_cpu_container],
        SimScalingConfiguration(),
        DeploymentRanking([training_cpu])
    )

    return resnet_fd


def prepare_function_deployments() -> List[SimFunctionDeployment]:
    resnet_inference_fd = prepare_resnet_inference_deployment()

    resnet_training_fd = prepare_resnet_training_deployment()

    return [resnet_inference_fd, resnet_training_fd]


def get_resnet50_inference_cpu_image_properties():
    return [
        ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='arm32'),
        ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='x86'),
        ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='aarch64')
    ]


def get_galileo_worker_image_properties():
    return [
        ImageProperties('galileo-worker', parse_size_string('23M'), arch='arm32'),
        ImageProperties('galileo-worker', parse_size_string('23M'), arch='x86'),
        ImageProperties('galileo-worker', parse_size_string('23M'), arch='aarch64')
    ]


def get_resnet50_training_cpu_image_properties():
    return [
        ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='arm32'),
        ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='x86'),
        ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='aarch64')
    ]


def prepare_client_deployments(ia_generator, client_names: List[str], deployment: SimFunctionDeployment,
                               max_requests: int, request_factory):
    client_fds = []
    for idx, client in enumerate(client_names):
        client_inference_fd = prepare_client_deployment(
            str(idx),
            client,
            ia_generator,
            max_requests,
            request_factory,
            deployment
        )
        client_fds.append(client_inference_fd)

    return client_fds


def prepare_inference_clients(clients: List[str], request_factory, deployment: SimFunctionDeployment) -> List[
    SimFunctionDeployment]:
    inference_max_rps = 5
    inference_max_requests = 200
    # generate profile
    inference_ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=inference_max_rps), max_ia=1)

    inference_client_fds = prepare_client_deployments(
        inference_ia_generator,
        clients,
        deployment, inference_max_requests,
        request_factory
    )
    return inference_client_fds


def prepare_training_clients(clients: List[str], request_factory, deployment: SimFunctionDeployment) -> List[
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
        deployment, training_max_requests,
        request_factory
    )

    return training_client_fds


class DecentralizedTrainInferenceBenchmark(Benchmark):

    def __init__(self, clients: List[Node], inference_request_factory: FunctionRequestFactory,
                 training_request_factory: FunctionRequestFactory):
        self.clients = clients
        self.inference_request_factory = inference_request_factory
        self.training_request_factory = training_request_factory

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        images = []
        resnet50_inference_cpu_img_properties = get_resnet50_inference_cpu_image_properties()
        resnet50_training_cpu_img_properties = get_resnet50_training_cpu_image_properties()
        galileo_worker_image_properties = get_galileo_worker_image_properties()

        images.extend(resnet50_inference_cpu_img_properties)
        images.extend(resnet50_training_cpu_img_properties)
        images.extend(galileo_worker_image_properties)

        for image in images:
            containers.put(image)

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        # deploy functions
        deployments = []
        fn_deployments = prepare_function_deployments()
        client_deployments = self.prepare_client_deployments_for_experiment(fn_deployments)

        deployments.extend(fn_deployments)
        deployments.extend(client_deployments)

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

    def prepare_client_deployments_for_experiment(self, deployments: List[SimFunctionDeployment]) -> List[
        SimFunctionDeployment]:

        fds = []
        inference_clients = [self.clients[0].name]
        training_clients = [self.clients[1].name]
        fds.extend(prepare_inference_clients(inference_clients, self.inference_request_factory, deployments[0]))
        fds.extend(prepare_training_clients(training_clients, self.training_request_factory, deployments[1]))

        return fds


def execute_benchmark():
    topology = testbed_topology()

    clients = find_clients(topology)

    #  inference factory - we assume that the file size is 250KB - client is a raspberry pi
    inference_factory = SimpleFunctionRequestFactory(size=250)

    # training factory - we assume that the file size is 10MB (100000KB) - client is a raspberry pi
    train_factory = SimpleFunctionRequestFactory(size=10000)

    benchmark = DecentralizedTrainInferenceBenchmark(clients, inference_factory, train_factory)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(topology, benchmark)
    sim.create_simulator_factory = ClientAIFunctionSimulatorFactory

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


def extract_dfs(sim):
    return {
        'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
        'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
        'traces_df': sim.env.metrics.extract_dataframe('traces'),
        'scale_df': sim.env.metrics.extract_dataframe('scale'),
        'schedule_df': sim.env.metrics.extract_dataframe('schedule'),
        'replica_deployment_df': sim.env.metrics.extract_dataframe('replica_deployment'),
        'function_deployments_df': sim.env.metrics.extract_dataframe('function_deployments'),
        'function_deployment_df': sim.env.metrics.extract_dataframe('function_deployment'),
        'function_deployment_lifecycle_df': sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
        'functions_df': sim.env.metrics.extract_dataframe('functions'),
        'flow_df': sim.env.metrics.extract_dataframe('flow'),
        'network_df': sim.env.metrics.extract_dataframe('network'),
        'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
        'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization'),
        'fets_df': sim.env.metrics.extract_dataframe('fets')
    }


def main():
    logging.basicConfig(level=logging.DEBUG)
    duration, sim = execute_benchmark()
    env = sim.env
    dfs = extract_dfs(sim)
    logger.info(f'Time passed in simulation: {env.now}, wall time passed: {duration}')
    logger.info('Mean exec time %d', dfs['invocations_df']['ts_exec'].mean())
    logger.info(f'Fets invocations: {len(dfs["fets_df"])}')


if __name__ == '__main__':
    main()
