import logging
import os
import sys
from typing import List, Dict
import joblib

from skippy.core.scheduler import Scheduler
from skippy.core.utils import parse_size_string

from ext.raith21.util import vanilla
from ext.tmueller23.functionsim import PowerPredictionSimulatorFactory, client_node_label, size_label
from ext.tmueller23.oracle import TMueller23FetOracle, TMueller23ResourceOracle
from ext.tmueller23.oracle.fet import ai_execution_time_distributions
from ext.tmueller23.oracle.resources import ai_resources_per_node_image
from ext.tmueller23.predicates import HostnamePredicate, hostname_label
from ext.tmueller23.topology.nano import NanoScenario
from ext.tmueller23.topology.nx import NxScenario
from ext.tmueller23.topology.xeongpu import XeonScenario
from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties, ContainerRegistry
from sim.faas import FunctionDeployment, Function, FunctionImage, ScalingConfiguration, \
    FunctionContainer, DefaultFaasSystem, FunctionCharacterization
from sim.faassim import Simulation
from sim.logging import RuntimeLogger, SimulatedClock
from sim.metrics import Metrics
from sim.oracle.oracle import ResourceOracle, FetOracle
from sim.requestgen import function_trigger, pre_recorded_profile
from sim.skippy import SimulationClusterContext
from sim.topology import Topology

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    nodename = 'nx'
    if len(sys.argv) > 1:
        nodename = sys.argv[1]

    scaling_factor = 1
    if len(sys.argv) > 2:
        scaling_factor = int(sys.argv[2])

    # a topology holds the cluster configuration and network topology
    if nodename == 'xeon':
        topology = xeon_topology(scaling_factor)
    elif nodename == 'nano':
        topology = nano_topology(scaling_factor)
    else:
        topology = nx_topology(scaling_factor)

    # a benchmark is a simpy process that sets up the runtime system (e.g., creates container images, deploys functions)
    # and creates workload by simulating function requests
    benchmark = PowerPredictionBenchmark()

    fet_oracle = TMueller23FetOracle(ai_execution_time_distributions)
    resource_oracle = TMueller23ResourceOracle(ai_resources_per_node_image)

    # Initialize environment
    env = Environment()

    env.simulator_factory = PowerPredictionSimulatorFactory(
        get_tmueller23_function_characterizations(resource_oracle, fet_oracle))
    env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
    env.topology = topology
    env.faas = DefaultFaasSystem(env, scale_by_requests=True)
    env.container_registry = ContainerRegistry()
    env.cluster = SimulationClusterContext(env)
    env.scheduler = configure_scheduler(env)

    sim = Simulation(env.topology, benchmark, env=env)
    result = sim.run()

    # print(sim.env.metrics.records)

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
        "utilization_df": sim.env.metrics.extract_dataframe('function_utilization'),
        "fets_df": sim.env.metrics.extract_dataframe('fets'),
        "function_replicas_df": sim.env.metrics.extract_dataframe('function_replicas'),
        "allocation_df": sim.env.metrics.extract_dataframe('allocation'),
        "simulation_duration_df": sim.env.metrics.extract_dataframe('simulation_duration')
    }

    for k, df in dfs.items():
        df.to_csv(f'data/output/{nodename}/efficientnet-inference-cpu/{k}.csv', index=False)

    # print(len(dfs))


def configure_scheduler(env):
    predicates = []
    predicates.extend(Scheduler.default_predicates)
    predicates.extend([HostnamePredicate()])

    priorities = vanilla.get_priorities()

    sched_params = {
        'percentage_of_nodes_to_score': 100,
        'priorities': priorities,
        'predicates': predicates
    }

    return Scheduler(env.cluster, **sched_params)


def xeon_topology(n=1) -> Topology:
    t = Topology()
    XeonScenario(n).materialize(t)
    t.init_docker_registry()

    return t


def nx_topology(n=1) -> Topology:
    t = Topology()
    NxScenario(n).materialize(t)
    t.init_docker_registry()

    return t


def nano_topology(n=1) -> Topology:
    t = Topology()
    NanoScenario(n).materialize(t)
    t.init_docker_registry()

    return t


class PowerPredictionBenchmark(Benchmark):

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        # populate the global container registry with images
        containers.put(ImageProperties('resi5/efficientnet-inference-cpu', parse_size_string('2640M'), arch='x86'))
        containers.put(ImageProperties('resi5/efficientnet-inference-cpu', parse_size_string('978M'), arch='aarch64'))

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

        set_power_prediction(env, 'data/tmueller23')

    def run(self, env: Environment):
        # deploy functions
        deployments = self.prepare_deployments(env)

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        # block until replicas become available (scheduling has finished and replicas have been deployed on the node)
        logger.info('waiting for replica')
        for deployment in deployments:
            yield env.process(
                env.faas.poll_available_replica(deployment.fn.name))

        # run profile
        ps = []
        for deployment in deployments:
            # generate profile
            ia_generator = pre_recorded_profile('data/tmueller23/interval.pkl')
            ps.append(function_trigger(env, deployment, ia_generator, max_requests=100))

        for p in ps:
            yield from p

    def prepare_deployments(self, env: Environment) -> List[FunctionDeployment]:
        deployments = []
        n = int((len(env.topology.get_nodes()) - 1) / 2)  # -1 because of registry node, /2 because of node/nuc tuples

        # logger.debug(f'DEBUG nodes: {env.topology.get_nodes()}')
        # logger.debug(f'DEBUG number of node tuples: {n}')

        for i in range(n):
            function_deployment = self.prepare_deployment(env, f'efficientnet-inference-{i}', 'resi5/efficientnet-inference-cpu', i)
            deployments.append(function_deployment)

        return deployments

    def prepare_deployment(self, env: Environment, function_name: str, function_image: str, i):
        # Design Time
        function_image = FunctionImage(image=function_image)
        function = Function(function_name, fn_images=[function_image])

        # Run time
        hostname = None
        for node in env.topology.get_nodes():
            if 'nx' in node.name or 'nano' in node.name or 'xeon' in node.name:
                if str(i) in node.name:
                    hostname = node.name

        client_name = None
        for node in env.topology.get_nodes():
            if 'nuc' in node.name:
                if str(i) in node.name:
                    client_name = node.name

        size = parse_size_string('636K')
        function_container = FunctionContainer(function_image,
                                               labels={'workers': '1',
                                                       hostname_label: hostname,
                                                       client_node_label: client_name,
                                                       size_label: size})

        function_deployment = FunctionDeployment(
            function,
            [function_container],
            ScalingConfiguration()
        )

        return function_deployment


def get_model_file(folder, node_name):
    if 'xeongpu' in node_name:
        file = 'eb-xeongpu.sav'
    elif 'nx' in node_name:
        file = 'eb-jetson-nx-01.sav'
    elif 'nano' in node_name:
        file = 'eb-jetson-nano-01.sav'
    else:
        raise ValueError(f"Can't find model for node: {node_name}")
    return os.path.join(folder, file)


def set_power_prediction(env: Environment, folder: str):
    models = {}
    for ether_node in env.topology.get_nodes():
        try:
            name = ether_node.name[:ether_node.name.rindex("_")]
            # logger.debug(f'Name: {name}')
            model = models.get(name, None)
            if model is None:
                model_file = get_model_file(folder, name)
                # logger.debug(f'Model file: {model_file}')
                model = joblib.load(model_file)
                models[name] = model

            env.power_models[ether_node.name] = model
        except ValueError:
            # happens when an ether_node has no '_', i.e. docker registry -> can be ignored
            pass


def get_tmueller23_function_characterizations(resource_oracle: ResourceOracle,
                                              fet_oracle: FetOracle) -> Dict[str, FunctionCharacterization]:
    return {
        'resi5/resnet-inference-cpu': FunctionCharacterization(
            'resi5/resnet-inference-cpu', fet_oracle, resource_oracle),
        'resi5/resnet-inference-gpu': FunctionCharacterization(
            'resi5/resnet-inference-gpu', fet_oracle, resource_oracle),
        'resi5/efficientnet-inference-cpu': FunctionCharacterization(
            'resi5/efficientnet-inference-cpu', fet_oracle, resource_oracle),
        'resi5/efficientnet-inference-gpu': FunctionCharacterization(
            'resi5/efficientnet-inference-gpu', fet_oracle, resource_oracle),
        'edgerun/objectdetection-cpu': FunctionCharacterization(
            'edgerun/objectdetection-cpu', fet_oracle, resource_oracle)
    }


if __name__ == '__main__':
    main()
