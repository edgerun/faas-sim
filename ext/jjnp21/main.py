#!/usr/bin/env python
# coding: utf-8
import logging
import random
from typing import Dict

import numpy as np
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex

from ext.jjnp21.load_balancers.localized_lrt import LocalizedLRTLBWrapper, LocalizedLeastResponseTimeLoadBalancer
from ext.jjnp21.load_balancers.localized_rr import LocalizedRoundRobinLoadBalancer, LocalizedRRLBWrapper
from ext.jjnp21.localized_lb_system import LocalizedLoadBalancerFaasSystem
from ext.jjnp21.topology import IndustrialIoTScenario, get_non_client_nodes
from ext.raith21 import images
from ext.raith21.benchmark.constant import ConstantBenchmark
from ext.raith21.characterization import get_raith21_function_characterizations
from ext.raith21.deployments import create_all_deployments
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.functionsim import AIPythonHTTPSimulatorFactory
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.oracles import Raith21ResourceOracle, Raith21FetOracle
from ext.raith21.predicates import CanRunPred, NodeHasAcceleratorPred, NodeHasFreeGpu, NodeHasFreeTpu
from ext.raith21.resources import ai_resources_per_node_image
from ext.raith21.topology import urban_sensing_topology, HeterogeneousUrbanSensingScenario
from ext.raith21.util import vanilla
from sim.core import Environment
from sim.docker import ContainerRegistry
from sim.faas.system import DefaultFaasSystem
from sim.faassim import Simulation
from sim.logging import SimulatedClock, RuntimeLogger
from sim.metrics import Metrics
from sim.skippy import SimulationClusterContext
from ether.vis import draw_basic
from ether.core import Node, Connection
from ether.cell import LANCell
import matplotlib.pyplot as plt
from ether.qos import latency
import pickle
from dataclasses import dataclass

from sim.topology import Topology


@dataclass(init=True)
class TestRunSettings():
    title: str
    lb_type: str
    placement_policy: str
    duration: int = 300
    req_per_sec: int = 50


def make_test_run(settings: TestRunSettings):
    np.random.seed(1234)
    random.seed(1234)
    logging.basicConfig(level=logging.INFO)

    num_devices = 100
    devices = generate_devices(num_devices, cloudcpu_settings)
    ether_nodes = convert_to_ether_nodes(devices)

    fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
    resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)

    deployments = list(create_all_deployments(fet_oracle, resource_oracle).values())
    function_images = images.all_ai_images

    predicates = []
    predicates.extend(Scheduler.default_predicates)
    predicates.extend([
        CanRunPred(fet_oracle, resource_oracle),
        NodeHasAcceleratorPred(),
        NodeHasFreeGpu(),
        NodeHasFreeTpu()
    ])

    priorities = vanilla.get_priorities()

    sched_params = {
        'percentage_of_nodes_to_score': 100,
        'priorities': priorities,
        'predicates': predicates
    }

    # Set arrival profiles/workload pattern
    benchmark = ConstantBenchmark('mixed', duration=settings.duration, rps=settings.req_per_sec)

    # Initialize topology
    storage_index = StorageIndex()
    # topology = urban_sensing_topology(ether_nodes, storage_index)
    topology = Topology()
    # HeterogeneousUrbanSensingScenario(ether_nodes, storage_index).materialize(topology)

    scenario1 = IndustrialIoTScenario('iot-1', num_premises=5, clients_per_premise=2, internet='internet_chix')
    scenario1.materialize(topology)
    scenario2 = IndustrialIoTScenario('iot-2', num_premises=2, clients_per_premise=4, internet='internet_nyc')
    scenario2.materialize(topology)
    topology.add_connection(Connection('internet_chix', 'internet_nyc', latency_dist=latency.business_isp))

    topology.init_docker_registry()
    # we should be able to attach something to the upstream 'switch_cloudlet_0'

    cental_lb_node = Node('load-balancer')
    topology.add_node(cental_lb_node)
    c = LANCell([cental_lb_node], backhaul='internet_chix')
    # c = LANCell([cental_lb_node], backhaul='internet')
    c.materialize(topology)

    all_lb_nodes = [node for node in get_non_client_nodes(topology) if isinstance(node, Node)]

    # draw_basic(topology)
    # plt.show()

    # Initialize environment
    env = Environment()

    env.simulator_factory = AIPythonHTTPSimulatorFactory(
        get_raith21_function_characterizations(resource_oracle, fet_oracle))
    env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
    env.topology = topology

    env.faas = LocalizedLoadBalancerFaasSystem(env, scale_by_queue_requests_per_replica=True)
    # env.faas = LocalizedLoadBalancerFaasSystem(env, scale_by_average_requests=True)
    lb_location = settings.placement_policy  # can be 'dist' | 'central'
    lb_type = settings.lb_type  # can be 'lrt' | 'rr'
    lb_resolver = 'closest'  # can be 'closest' | 'random'

    # somewhat important ideas:
    # in some scenarios more requests come in than can be managed
    # eventually say after a time of 2000-3000ms execution should stop with a timeout error

    lb_nodes = []

    if lb_location == 'dist':
        lb_nodes = all_lb_nodes
    elif lb_location == 'central':
        lb_nodes = [cental_lb_node]

    load_balancers = []
    if lb_type == 'lrt':
        for node in lb_nodes:
            load_balancers.append(LocalizedLeastResponseTimeLoadBalancer(env, node, env.faas.replicas))
    elif lb_type == 'rr':
        for node in lb_nodes:
            load_balancers.append(LocalizedRoundRobinLoadBalancer(env, node, env.faas.replicas))

    wrapper_lb = None
    if lb_resolver == 'closest':
        wrapper_lb = LocalizedLRTLBWrapper(load_balancers, env)
    elif lb_resolver == 'random':
        wrapper_lb = LocalizedRRLBWrapper(load_balancers)

    env.faas.set_load_balancer(wrapper_lb)

    env.container_registry = ContainerRegistry()
    env.storage_index = storage_index
    env.cluster = SimulationClusterContext(env)
    env.scheduler = Scheduler(env.cluster, **sched_params)

    sim = Simulation(env.topology, benchmark, env=env)
    result = sim.run()

    dfs = {
        "invocations_df": sim.env.metrics.extract_dataframe('invocations'),
        "scale_df": sim.env.metrics.extract_dataframe('scale'),
        "schedule_df": sim.env.metrics.extract_dataframe('schedule'),
        "replica_deployment_df": sim.env.metrics.extract_dataframe('replica_deployment'),
        "function_deployments_df": sim.env.metrics.extract_dataframe('function_deployments'),
        "function_deployment_df": sim.env.metrics.extract_dataframe('function_deployment'),
        "function_deployment_lifecycle_df": sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
        "functions_df": sim.env.metrics.extract_dataframe('functions'),
        "flow_df": sim.env.metrics.extract_dataframe('flow'),
        "network_df": sim.env.metrics.extract_dataframe('network'),
        "utilization_df": sim.env.metrics.extract_dataframe('utilization'),
        'fets_df': sim.env.metrics.extract_dataframe('fets')
    }
    return dfs


def make_analysis(dfs, title: str = 'unknown experiment'):
    print('***********************')
    print(title)
    print('***********************')
    inv = dfs['invocations_df']
    # inv = inv[inv['function_name'] == 'speech-inference']
    # inv = inv[inv['function_name'] == 'mobilenet-inference']
    # inv = inv[inv['function_name'] == 'resnet50-inference']

    plt.plot(inv.index, inv['t_exec'].rolling('1s').mean())
    # plt.plot(inv.index, inv['t_exec'])
    plt.plot(inv.index, inv['t_exec'].rolling('1s').count())
    plt.plot(inv.index, inv['lb_client_latency'].rolling('1s').count())
    plt.plot(inv.index, inv['lb_function_latency'].rolling('1s').count())
    # plt.plot(inv.index, inv['t_exec'].rolling('5s').count() / 5)
    print('mean: ' + str(inv['t_exec'].mean()))
    print('median: ' + str(inv['t_exec'].quantile(0.5)))
    print('q90: ' + str(inv['t_exec'].quantile(0.9)))
    print('q99: ' + str(inv['t_exec'].quantile(0.99)))
    print('lb_client_latency: ' + str(inv['lb_client_latency'].mean()))
    print('lb_function_latency: ' + str(inv['lb_function_latency'].mean()))
    fet = dfs['fets_df']
    print('avg FET: ' + str(fet['t_fet'].mean()))
    print('median FET: ' + str(fet['t_fet'].quantile(0.5)))
    print('q90 FET: ' + str(fet['t_fet'].quantile(0.9)))
    print('q99 FET: ' + str(fet['t_fet'].quantile(0.99)))
    print('avg wait: ' + str(fet['t_wait'].mean()))
    print('median wait: ' + str(fet['t_wait'].quantile(0.5)))
    print('q90 wait: ' + str(fet['t_wait'].quantile(0.9)))
    print('q99 wait: ' + str(fet['t_wait'].quantile(0.99)))
    print('total requests sent/received: ' + str(inv['t_exec'].count()))
    print('TX time avg ' + str(inv['tx_time'].mean()))
    print('TX time q50 ' + str(inv['tx_time'].quantile(0.5)))
    print('TX time q90 ' + str(inv['tx_time'].quantile(0.9)))
    print('TX time q99 ' + str(inv['tx_time'].quantile(0.99)))
    plt.show()
    nodes = inv['node'].unique()
    types = ['rpi3', 'tx2', 'nuc', 'rockpi', 'rpi4', ]
    results = {}
    node_counts = {}
    typed_results = {}
    avg_typed_results = {}
    for n in nodes:
        cnt = inv[inv['node'] == n]['node'].count()
        results[n] = cnt
    for t in types:
        typed_results[t] = 0
        node_counts[t] = 0
        for n, cnt in results.items():
            if n.startswith(t):
                typed_results[t] += cnt
                node_counts[t] += 1

    for t, cnt in typed_results.items():
        avg_typed_results[t] = cnt / max(node_counts[t], 1)
    print('Avg requests per node type:')
    print(avg_typed_results)
    print('Present node types')
    print(node_counts)

    # with open('outfile.pickle', 'wb') as f:
    #     pickle.dump(dfs, f)
    #     f.flush()
    #     f.close()
    # print(len(dfs))
