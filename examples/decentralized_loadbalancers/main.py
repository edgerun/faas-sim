import logging
import time
from typing import List, Callable, Any

from faas.system import FunctionContainer
from faas.util.constant import controller_role_label, zone_label
from skippy.core.scheduler import Scheduler

from examples.decentralized_clients.main import get_resnet50_inference_cpu_image_properties, \
    get_resnet50_training_cpu_image_properties, get_galileo_worker_image_properties, \
    prepare_function_deployments
from examples.decentralized_loadbalancers.deployments import prepare_client_deployments_for_experiment, \
    prepare_load_balancer_deployments, get_go_load_balancer_image_props
from examples.decentralized_loadbalancers.topology import testbed_topology
from examples.watchdogs.inference import InferenceFunctionSim
from examples.watchdogs.training import TrainingFunctionSim
from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.faas import FunctionSimulator, SimulatorFactory
from sim.faas.core import Node, LocalizedSimRoundRobinBalancer, GlobalSimRoundRobinLoadBalancer
from sim.faas.loadbalancers import ForwardingClientSimulator, LoadBalancerSimulator, LoadBalancerOptimizerUpdateProcess
from sim.faassim import Simulation
from sim.predicates import PodHostEqualsNode
from sim.requestgen import SimpleFunctionRequestFactory
from sim.util.client import find_clients
from sim.util.experiment import save_results, extract_dfs
from sim.util.loadbalancer import find_lbs

logger = logging.getLogger(__name__)


def create_scheduler(env: Environment):
    scheduler = Scheduler(env.cluster)
    scheduler.predicates.append(PodHostEqualsNode())
    # scheduler.predicates.append(CheckNodeLabelPresencePred([client_label]))
    return scheduler


class DecentralizedAIFunctionSimulatorFactory(SimulatorFactory):

    def __init__(self, load_balancer_factory: Callable[[Environment, FunctionContainer], Any]):
        self.load_balancer_factory = load_balancer_factory

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        if 'inference' in fn.fn_image.image:
            return InferenceFunctionSim(4)
        elif 'training' in fn.fn_image.image:
            return TrainingFunctionSim()
        elif 'galileo-worker' in fn.fn_image.image:
            return ForwardingClientSimulator()
        elif 'load-balancer' == fn.fn_image.image:
            return LoadBalancerSimulator(self.load_balancer_factory(env, fn))


class DecentralizedLoadBalancerTrainInferenceBenchmark(Benchmark):
    def __init__(self, clients: List[Node], load_balancers_hosts: List[Node]):
        self.balancer = 'load-balancer'
        self.clients = clients
        self.load_balancer_hosts = load_balancers_hosts
        self.metadata = {'benchmark': 'DecentralizedLoadBalancerTrainInferenceBenchmark'}

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
        load_balancer_deployment = prepare_load_balancer_deployments('load-balancer', self.load_balancer_hosts)
        client_deployments = prepare_client_deployments_for_experiment(self.clients, load_balancer_deployment,
                                                                       function_deployments)
        self.metadata['function_deployments'] = [f.name for f in function_deployments]
        self.metadata['client_deployments'] = [f.name for f in client_deployments]
        self.metadata['load_balancer_deployments'] = [f.name for f in load_balancer_deployment]

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




def execute_benchmark():
    topology = testbed_topology()

    clients = find_clients(topology)

    lbs = find_lbs(topology)
    benchmark = DecentralizedLoadBalancerTrainInferenceBenchmark(clients, lbs)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(topology, benchmark)
    lb_process = LoadBalancerOptimizerUpdateProcess(reconcile_interval=5)
    balancer = 'localized-rr-balancer'

    def create_load_balancer(env: Environment, fn: FunctionContainer):
        if 'rr-balancer' == balancer:
            return GlobalSimRoundRobinLoadBalancer(env)
        elif 'localized-rr-balancer' == balancer:
            cluster = fn.labels[zone_label]
            return LocalizedSimRoundRobinBalancer(env, cluster)

    # override the SimulatorFactory factory
    sim.env.benchmark = benchmark
    sim.env.topology = topology
    sim.init_environment(sim.env)
    sim.env.scheduler = create_scheduler(sim.env)
    sim.env.simulator_factory = DecentralizedAIFunctionSimulatorFactory(create_load_balancer)
    sim.env.background_processes.append(lb_process.run)
    # run the simulation
    start = time.time()
    sim.run()
    end = time.time()
    duration = end - start

    return duration, sim


def main():
    logging.basicConfig(level=logging.DEBUG)
    logger.info('Start decentralized load balancers example.')
    root_folder = 'results'
    duration, sim = execute_benchmark()
    env = sim.env
    dfs = extract_dfs(sim)

    logger.info(f'Time passed in simulation: {env.now}, wall time passed: {duration}')
    logger.info('Mean exec time %d', dfs['invocations_df']['t_exec'].mean())
    logger.info(f'Fets invocations: {len(dfs["fets_df"])}')

    logger.info(f'Saving results')
    results = save_results(root_folder, dfs, sim)
    logger.info(f'Results saved under {results}')

    logger.info('End decentralized load balancers example.')


if __name__ == '__main__':
    main()
