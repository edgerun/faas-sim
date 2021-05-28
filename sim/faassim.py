import logging
import time

from skippy.core.scheduler import Scheduler

from sim.benchmark import Benchmark
from sim.core import Environment, timeout_listener
from sim.docker import ContainerRegistry, pull as docker_pull
from sim.faas import FunctionReplica, FunctionRequest, FunctionSimulator, SimulatorFactory, FunctionContainer
from sim.faas.system import DefaultFaasSystem
from sim.metrics import Metrics, RuntimeLogger
from sim.resource import MetricsServer, ResourceState, ResourceMonitor
from sim.skippy import SimulationClusterContext
from sim.topology import Topology

logger = logging.getLogger(__name__)


class BadPlacementException(BaseException):
    pass


class Simulation:

    def __init__(self, topology: Topology, benchmark: Benchmark, env: Environment = None, timeout=None, name=None):
        self.env = env or Environment()
        self.topology = topology
        self.benchmark = benchmark
        self.timeout = timeout
        self.name = name

    def run(self):
        logger.info('initializing simulation, benchmark: %s, topology nodes: %d',
                    type(self.benchmark).__name__, len(self.topology.nodes))

        env = self.env

        env.benchmark = self.benchmark
        env.topology = self.topology

        self.init_environment(env)

        then = time.time()

        if self.timeout:
            logger.info('starting timeout listener with timeout %d', self.timeout)
            env.process(timeout_listener(env, then, self.timeout))

        logger.info('starting resource monitor')
        env.process(env.resource_monitor.run())

        logger.info('setting up benchmark')
        self.benchmark.setup(env)

        logger.info('starting faas system')
        env.faas.start()

        logger.info('starting benchmark process')
        p = env.process(self.benchmark.run(env))

        logger.info('executing simulation')
        env.run(until=p)

        logger.info('simulation ran %.2fs sim, %.2fs wall', env.now, (time.time() - then))

    def init_environment(self, env):
        if not env.simulator_factory:
            env.simulator_factory = env.simulator_factory or self.create_simulator_factory()

        if not env.container_registry:
            env.container_registry = self.create_container_registry()

        if not env.faas:
            env.faas = self.create_faas_system(env)

        if not env.metrics:
            env.metrics = Metrics(env, RuntimeLogger())

        if not env.cluster:
            env.cluster = SimulationClusterContext(env)

        if not env.scheduler:
            env.scheduler = self.create_scheduler(env)

        if not env.metrics_server:
            env.metrics_server = MetricsServer()

        if not env.resource_state:
            env.resource_state = ResourceState()

        if not env.resource_monitor:
            env.resource_monitor = ResourceMonitor(env, 1)

    def create_container_registry(self):
        return ContainerRegistry()

    def create_simulator_factory(self):
        return SimpleSimulatorFactory()

    def create_faas_system(self, env):
        return DefaultFaasSystem(env)

    def create_scheduler(self, env):
        return Scheduler(env.cluster)


class DummySimulator(FunctionSimulator):

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def startup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def setup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        yield env.timeout(0)

    def teardown(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)


class DockerDeploySimMixin:
    def deploy(self, env: Environment, replica: FunctionReplica):
        yield from docker_pull(env, replica.image, replica.node.ether_node)


class ModeledExecutionSimMixin:

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        # 1) get parameters of base distribution (ideal case)
        # 2) check the utilization of the node the replica is running on
        # 3) transform distribution parameters with degradation function depending on utilization
        # 4) sample from that distribution
        logger.info('invoking %s on %s (%d in parallel)', request.name, replica.node.name,
                    len(replica.node.current_requests))

        yield env.timeout(1)


class SimpleFunctionSimulator(ModeledExecutionSimMixin, DockerDeploySimMixin, DummySimulator):
    pass


class SimpleSimulatorFactory(SimulatorFactory):
    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        return SimpleFunctionSimulator()
