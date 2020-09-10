import logging
import time

import sim.docker as docker
from sim.benchmark import Benchmark
from sim.core import Environment, timeout_listener
from sim.faas import FaasSystem, FunctionReplica, FunctionRequest
from sim.topology import Topology

logger = logging.getLogger(__name__)


class BadPlacementException(BaseException):
    pass


class Simulation:

    def __init__(self, env: Environment, benchmark: Benchmark, topology: Topology, timeout=None, name=None):
        self.env = env
        self.benchmark = benchmark
        self.topology = topology
        self.timeout = timeout
        self.name = name

    def run(self):
        logger.info('initializing simulation, benchmark: %s, topology nodes: %d',
                    type(self.benchmark).__name__, len(self.topology.nodes))

        env = self.env

        env.benchmark = self.benchmark
        env.topology = self.topology
        env.faas = FaasSystem(env)
        env.registry = docker.Registry()

        then = time.time()

        if self.timeout:
            logger.info('starting timeout listener with timeout %d', self.timeout)
            env.process(timeout_listener(env, then, self.timeout))

        logger.info('setting up benchmark')
        self.benchmark.setup(env)

        logger.info('starting benchmark process')
        p = env.process(self.benchmark.run(env))

        logger.info('executing simulation')
        env.run(until=p)

        logger.info('simulation ran %.2fs sim, %.2fs wall', env.now, (time.time() - then))


class FunctionSimulator:

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def startup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def execute(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        yield env.timeout(0)

    def teardown(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

