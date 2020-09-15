import abc
import enum
import logging
import time
from collections import defaultdict
from typing import List, Dict, NamedTuple

import simpy
from ether.util import parse_size_string

from sim.core import Environment
from sim.skippy import create_function_pod
from skippy.core.model import Pod

logger = logging.getLogger(__name__)


def counter(start: int = 1):
    n = start
    while True:
        yield n
        n += 1


class FunctionState(enum.Enum):
    CONCEIVED = 1
    STARTING = 2
    RUNNING = 3
    SUSPENDED = 4


class Resources:
    memory: int
    cpu: int

    def __init__(self, cpu_millis: int = 1 * 1000, memory: int = 1024 * 1024 * 1024):
        self.memory = memory
        self.cpu = cpu_millis

    def __str__(self):
        return 'Resources(CPU: {0} Memory: {1})'.format(self.cpu, self.memory)

    @staticmethod
    def from_str(memory, cpu):
        """
        :param memory: "64Mi"
        :param cpu: "250m"
        :return:
        """
        return Resources(parse_size_string(memory), int(cpu.rstrip('m')))


class FunctionDefinition:
    name: str
    image: str
    labels: Dict[str, str] = {}

    requests: Resources = Resources()

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 20
    scale_zero: bool = False

    def __init__(self, name: str, image: str) -> None:
        super().__init__()
        self.name = name
        self.image = image


class FunctionReplica:
    """
    A function replica is an instance of a function running on a specific node.
    """
    function: FunctionDefinition
    node: str
    pod: Pod
    state: FunctionState = FunctionState.CONCEIVED


class FunctionRequest:
    request_id: int
    name: str
    size: float = None

    id_generator = counter()

    def __init__(self, name, size=None) -> None:
        super().__init__()
        self.name = name
        self.size = size
        self.request_id = next(self.id_generator)

    def __str__(self) -> str:
        return 'FunctionRequest(%d, %s, %s)' % (self.request_id, self.name, self.size)

    def __repr__(self):
        return self.__str__()


class FunctionResponse(NamedTuple):
    request_id: int
    code: int
    t_wait: float = 0
    t_exec: float = 0
    node: str = None


class FaasSystem:
    """

    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.functions = dict()
        self.replicas = defaultdict(list)

        self.request_queue = simpy.Store(env)
        self.scheduler_queue = simpy.Store(env)

    def get_replicas(self, fn_name: str, state=None) -> List[FunctionReplica]:
        if state is None:
            return self.replicas[fn_name]

        return [replica for replica in self.replicas[fn_name] if replica.state == state]

    def deploy(self, fn: FunctionDefinition):
        if fn.name in self.functions:
            raise ValueError('function already deployed')

        self.functions[fn.name] = self.functions

        replica = FunctionReplica()
        replica.function = fn
        replica.pod = self.create_pod(fn)

        yield self.scheduler_queue.put(replica)

    def request(self, request: FunctionRequest):
        # TODO
        logger.debug('invoking function %s', request.name)
        yield self.env.timeout(0)

    def create_pod(self, fn: FunctionDefinition):
        return create_function_pod(fn)

    def start(self):
        self.env.process(self.run_scheduler_worker())

    def run_scheduler_worker(self):
        env = self.env

        while True:
            replica: FunctionReplica
            replica = yield self.scheduler_queue.get()

            logger.debug('scheduling next replica %s', replica.function.name)

            # schedule the required pod
            pod = replica.pod
            then = time.time()
            result = env.scheduler.schedule(pod)
            duration = time.time() - then
            yield env.timeout(duration)  # include scheduling latency in simulation time

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Pod scheduling took %.2f ms, and yielded %s', duration * 1000, result)

            if not result.suggested_host:
                logger.error('pod %s cannot be scheduled', pod.name)
                continue

            logger.info('pod %s was scheduled to %s', pod.name, result.suggested_host)

            replica.node = result.suggested_host.name

            # start a new process to simulate starting of pod
            env.process(simulate_function_start(env, replica))


def simulate_function_start(env: Environment, replica: FunctionReplica):
    # TODO: registry of simulators?
    sim: FunctionSimulator = env.simulator_factory.create(env, replica.function)

    logger.debug('deploying function %s to %s', replica.function.name, replica.node)
    yield from sim.deploy(env, replica)
    replica.state = FunctionState.STARTING
    logger.debug('starting function %s on %s', replica.function.name, replica.node)
    yield from sim.startup(env, replica)

    logger.debug('running function setup %s on %s', replica.function.name, replica.node)
    replica.state = FunctionState.RUNNING
    yield from sim.setup(env, replica)  # FIXME: this is really domain-specific startup


class FunctionSimulator(abc.ABC):

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def startup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def setup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)

    def execute(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        yield env.timeout(0)

    def teardown(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)


class SimulatorFactory:

    def create(self, env: Environment, fn: FunctionDefinition) -> FunctionSimulator:
        raise NotImplementedError
