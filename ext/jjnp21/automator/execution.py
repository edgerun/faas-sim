import logging
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.experiments.net_vis import draw_custom
from ext.jjnp21.load_balancers.localized_lrt import LocalizedLeastResponseTimeLoadBalancer, LocalizedLRTLBWrapper
from ext.jjnp21.load_balancers.localized_rr import LocalizedRoundRobinLoadBalancer
from ext.jjnp21.localized_lb_system import LocalizedLoadBalancerFaasSystem
from ext.jjnp21.topology import get_non_client_nodes
from ext.raith21.characterization import get_raith21_function_characterizations
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.functionsim import AIPythonHTTPSimulatorFactory
from ext.raith21.oracles import Raith21FetOracle, Raith21ResourceOracle
from ext.raith21.predicates import CanRunPred, NodeHasAcceleratorPred, NodeHasFreeGpu, NodeHasFreeTpu
from ext.raith21.resources import ai_resources_per_node_image
from ext.raith21.util import vanilla
from sim.core import Environment, Node
from ether.core import Connection
from sim.docker import ContainerRegistry
from sim.faassim import Simulation
from sim.logging import RuntimeLogger, SimulatedClock
from sim.metrics import Metrics
from ether.cell import LANCell
from sim.skippy import SimulationClusterContext
from skippy.core.scheduler import Scheduler


def extract_result_from_sim(sim: Simulation, exp: Experiment, run_duration_seconds: float = 0) -> Result:
    result = Result()
    result.experiment = exp
    result.run_duration_seconds = run_duration_seconds
    result.invocations = sim.env.metrics.extract_dataframe('invocations')
    result.scale = sim.env.metrics.extract_dataframe('scale')
    result.schedule = sim.env.metrics.extract_dataframe('schedule')
    result.replica_deployment = sim.env.metrics.extract_dataframe('replica_deployment')
    result.function_deployments = sim.env.metrics.extract_dataframe('function_deployments')
    result.function_deployment = sim.env.metrics.extract_dataframe('function_deployment')
    result.function_deployment_lifecycle = sim.env.metrics.extract_dataframe('function_deployment_lifecycle')
    result.functions = sim.env.metrics.extract_dataframe('functions')
    result.flow = sim.env.metrics.extract_dataframe('flow')
    result.network = sim.env.metrics.extract_dataframe('network')
    result.utilization = sim.env.metrics.extract_dataframe('utilization')
    result.fets = sim.env.metrics.extract_dataframe('fets')
    return result


def run_experiment(experiment: Experiment) -> Result:
    random.seed(experiment.seed)
    np.random.seed(experiment.seed)
    logging.basicConfig(level=logging.INFO)
    topology = experiment.topology_factory.create()
    # print('drawing...')
    # draw_custom(topology)
    # plt.show()
    # print('done drawing')
    benchmark = experiment.benchmark_factory.create()
    env = Environment()
    env.topology = topology
    env.benchmark = benchmark

    fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
    resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
    env.simulator_factory = AIPythonHTTPSimulatorFactory(
        get_raith21_function_characterizations(resource_oracle, fet_oracle))
    env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))


    # Configure scaling settings based on the experiment parametrization
    # Todo I could consider moving this part. On one hand it uses quite a few layers, but on the other hand
    # it keeps the "interpretation" of the experiment settings to one place
    faas_kwargs = {
        'scale_by_average_requests': experiment.function_scaling_strategy == FunctionScalingStrategy.AVG_REQUEST_RATE,
        'scale_by_queue_requests_per_replica': experiment.function_scaling_strategy == FunctionScalingStrategy.AVG_QUEUE_LENGTH,
        'scale_static': experiment.function_scaling_strategy == FunctionScalingStrategy.CUSTOM_STATIC,
        'net_mode': experiment.net_mode,
        'lb_scaler_factory': experiment.lb_scaler_factory
    }
    experiment.faas_system_factory.set_constructor_args(**faas_kwargs)

    faas = experiment.faas_system_factory.create(env)
    env.faas = faas

    # Load balancer shenanigans
    # central_lb_node = Node('load-balancer')
    # topology.add_node(central_lb_node)
    # c = LANCell([cental_lb_node], backhaul='internet_chix')
    central_lb_node = topology.get_load_balancer_node()
    print(f'central lb node is:{central_lb_node.name}')

    # topology.connect_load_balancer(central_lb_node)

    # todo: remove this once we are sure that the new system works properly
    # all_lb_nodes = [node for node in get_non_client_nodes(topology) if isinstance(node, Node) and node.name != 'registry' and not node.name.startswith('internet')]
    #
    # lb_nodes = []
    # if experiment.lb_placement_strategy == LoadBalancerPlacementStrategy.ALL_NODES:
    #     lb_nodes = all_lb_nodes
    # elif experiment.lb_placement_strategy == LoadBalancerPlacementStrategy.CENTRAL:
    #     lb_nodes = [central_lb_node]
    # else:
    #     raise Exception('Invalid load balancer placement strategy')
    #
    # load_balancers = []
    # if experiment.lb_type == LoadBalancerType.LEAST_RESPONSE_TIME:
    #     for node in lb_nodes:
    #         load_balancers.append(LocalizedLeastResponseTimeLoadBalancer(env, node, env.faas.replicas))
    # elif experiment.lb_type == LoadBalancerType.ROUND_ROBIN:
    #     for node in lb_nodes:
    #         load_balancers.append(LocalizedRoundRobinLoadBalancer(env, node, env.faas.replicas))
    # else:
    #     raise Exception('Invalid load balancer type')
    #
    # wrapper_lb = None
    # if experiment.client_lb_resolving_strategy == ClientLoadBalancerResolvingStrategy.LOWEST_PING:
    #     wrapper_lb = LocalizedLRTLBWrapper(load_balancers, env)
    # else:
    #     raise Exception('Invalid client load balancer resolution strategy')
    #
    # # in case we use a faas-system that allows for custom load balancers
    # if isinstance(env.faas, LocalizedLoadBalancerFaasSystem):
    #     env.faas.set_load_balancer(wrapper_lb)

    env.container_registry = ContainerRegistry()

    # env.storage_index = storage_index
    env.cluster = SimulationClusterContext(env)
    env.scheduler = experiment.function_scheduler_factory.create(env)

    # lb-scheduler things
    env.lb_scheduler = experiment.lb_scheduler_factory.create(env)

    sim = Simulation(env.topology, benchmark, env=env)
    start = time.time()
    sim.run()
    end = time.time()
    return extract_result_from_sim(sim, experiment, round(end - start, 2))
