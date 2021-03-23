#!/usr/bin/env python
# coding: utf-8
import logging
import random

import numpy as np
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex

from ext.raith21 import images
from ext.raith21.benchmark.constant import ConstantBenchmark
from ext.raith21.deployments import create_all_deployments
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.functionsim import AIPythonHTTPSimulatorFactory
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.oracles import Raith21ResourceOracle, Raith21FetOracle
from ext.raith21.predicates import CanRunPred, NodeHasAcceleratorPred, NodeHasFreeGpu, NodeHasFreeTpu
from ext.raith21.resources import ai_resources_per_node_image
from ext.raith21.topology import urban_sensing_topology
from ext.raith21.util import vanilla
from sim.core import Environment
from sim.docker import ContainerRegistry
from sim.faas.system import DefaultFaasSystem
from sim.faassim import Simulation
from sim.logging import SimulatedClock, RuntimeLogger
from sim.metrics import Metrics
from sim.skippy import SimulationClusterContext

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
benchmark = ConstantBenchmark('mixed', duration=200, rps=50)

# Initialize topology
storage_index = StorageIndex()
topology = urban_sensing_topology(ether_nodes, storage_index)

# Initialize environment
env = Environment()

env.simulator_factory = AIPythonHTTPSimulatorFactory()
env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
env.topology = topology
env.faas = DefaultFaasSystem(env, scale_by_requests=True)
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
print(len(dfs))
