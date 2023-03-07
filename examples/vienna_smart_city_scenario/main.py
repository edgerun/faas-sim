import logging
import multiprocessing
import random
import sys
import time
from typing import List, Callable, Any

import numpy as np
import srds
from faas.system import FunctionContainer, Metrics, NullLogger
from faas.util.constant import zone_label, client_role_label, worker_role_label, controller_role_label
from skippy.core.scheduler import Scheduler

from examples.vienna_smart_city_scenario.deployments import prepare_function_deployments, \
    get_go_load_balancer_image_props, get_resnet50_inference_cpu_image_properties, get_galileo_worker_image_properties, \
    prepare_load_balancer_deployments, prepare_client_deployments_for_experiment
from examples.vienna_smart_city_scenario.predicates import CheckNodeLabelPresencePred, ExclusivePred, LoadBalancerPred
from examples.vienna_smart_city_scenario.topology import vienna_smart_city_topology
from examples.watchdogs.inference import InferenceFunctionSim
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.oracles import Raith21FetOracle, Raith21ResourceOracle
from ext.raith21.resources import ai_resources_per_node_image
from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.faas import FunctionSimulator, SimulatorFactory
from sim.faas.core import Node, LocalizedSimRoundRobinBalancer, GlobalSimRoundRobinLoadBalancer
from sim.faas.loadbalancers import ForwardingClientSimulator, LoadBalancerSimulator, LoadBalancerOptimizerUpdateProcess
from sim.faassim import Simulation
from sim.metrics import SimMetrics
from sim.oracle.oracle import ResourceOracle, FetOracle
from sim.predicates import PodHostEqualsNode
from sim.requestgen import SimpleFunctionRequestFactory
from sim.util.client import find_clients
from sim.util.experiment import save_results, extract_dfs
from sim.util.loadbalancer import find_lbs

logger = logging.getLogger(__name__)


def create_scheduler(env: Environment):
    scheduler = Scheduler(env.cluster)
    scheduler.predicates.append(PodHostEqualsNode())
    scheduler.predicates.append(CheckNodeLabelPresencePred([worker_role_label]))
    scheduler.predicates.append(CheckNodeLabelPresencePred([client_role_label]))
    scheduler.predicates.append(LoadBalancerPred())
    scheduler.predicates.append(ExclusivePred())
    return scheduler


class DecentralizedAIFunctionSimulatorFactory(SimulatorFactory):

    def __init__(self, load_balancer_factory: Callable[[Environment, FunctionContainer], Any], fet_oracle: FetOracle,
                 resource_oracle: ResourceOracle):
        self.fet_oracle = fet_oracle
        self.resource_oracle = resource_oracle
        self.load_balancer_factory = load_balancer_factory

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        if 'inference' in fn.fn_image.image:
            return InferenceFunctionSim(self.fet_oracle, self.resource_oracle)
        elif 'galileo-worker' in fn.fn_image.image:
            return ForwardingClientSimulator()
        elif 'load-balancer' == fn.fn_image.image:
            return LoadBalancerSimulator(self.load_balancer_factory(env, fn))


class ViennaSmartCityBenchmark(Benchmark):
    def __init__(self, max_rps, max_requests, districts: int, resnet_inference_scale_min: int, clients: List[Node],
                 load_balancers_hosts: List[Node]):
        self.max_rps = max_rps
        self.max_requests = max_requests
        self.resnet_inference_scale_min = resnet_inference_scale_min
        self.balancer = 'load-balancer'
        self.clients = clients
        self.load_balancer_hosts = load_balancers_hosts
        self.metadata = {'max_rps': max_rps, 'max_requests': max_requests, 'benchmark': 'ViennaSmartCityBenchmark',
                         'resnet_inference_scale_min': resnet_inference_scale_min, 'districts': districts}

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        images = []
        resnet50_inference_cpu_img_properties = get_resnet50_inference_cpu_image_properties()
        galileo_worker_image_properties = get_galileo_worker_image_properties()
        lb_image_properties = get_go_load_balancer_image_props(self.balancer)

        images.extend(resnet50_inference_cpu_img_properties)
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
        function_deployments = prepare_function_deployments(self.resnet_inference_scale_min)
        load_balancer_deployment = prepare_load_balancer_deployments('load-balancer', self.load_balancer_hosts)
        client_deployments = prepare_client_deployments_for_experiment(self.max_rps, self.max_requests, self.clients,
                                                                       load_balancer_deployment,
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


def execute_benchmark(max_rps: int, max_requests: int, districts: int, resnet_inference_scale_min: int):
    topology = vienna_smart_city_topology(districts)

    clients = find_clients(topology)

    lbs = find_lbs(topology)
    benchmark = ViennaSmartCityBenchmark(max_rps, max_requests, districts, resnet_inference_scale_min, clients, lbs)

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
    sim.env.metrics = SimMetrics(sim.env, None)
    fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
    resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
    sim.env.simulator_factory = DecentralizedAIFunctionSimulatorFactory(create_load_balancer, fet_oracle,
                                                                        resource_oracle)
    sim.env.background_processes.append(lb_process.run)
    # run the simulation
    start = time.time()
    sim.run()
    end = time.time()
    duration = end - start

    return duration, sim


def main():
    random.seed(3838)
    np.random.seed(3838)
    srds.seed(3838)

    logging.basicConfig(level=logging.INFO)
    logger.info('Start decentralized load balancers example.')
    root_folder = 'results'
    resnet_inference_scale_min = 30
    max_rps = 100
    max_requests = int(sys.argv[1])
    districts = 15
    duration, sim = execute_benchmark(max_rps, max_requests, districts, resnet_inference_scale_min)
    env = sim.env
    dfs = extract_dfs(sim)

    logger.info(f'Time passed in simulation: {env.now}, wall time passed: {duration}')
    logger.info('Mean exec time %d', dfs['invocations_df']['t_exec'].mean())
    logger.info(f'Fets invocations: {len(dfs["fets_df"])}')

    logger.info(f'Saving results')
    # results = save_results(root_folder, dfs, sim)
    # logger.info(f'Results saved under {results}')

    logger.info('End decentralized load balancers example.')



if __name__ == '__main__':
    main()
