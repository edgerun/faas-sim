import abc
import enum
import logging
import time
from collections import defaultdict
from typing import List, Dict, NamedTuple

import simpy
from ether.util import parse_size_string

from sim.core import Environment, NodeState
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
    node: NodeState
    pod: Pod
    state: FunctionState = FunctionState.CONCEIVED

    simulator: 'FunctionSimulator' = None


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


class LoadBalancer:
    env: Environment
    replicas: Dict[str, List[FunctionReplica]]

    def __init__(self, env, replicas) -> None:
        super().__init__()
        self.env = env
        self.replicas = replicas

    def get_running_replicas(self, function: str):
        return [replica for replica in self.replicas[function] if replica.state == FunctionState.RUNNING]

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        raise NotImplementedError


class RoundRobinLoadBalancer(LoadBalancer):

    def __init__(self, env, replicas) -> None:
        super().__init__(env, replicas)
        self.counters = defaultdict(lambda: 0)

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        replicas = self.get_running_replicas(request.name)
        i = self.counters[request.name] % len(replicas)
        self.counters[request.name] = (i + 1) % len(replicas)

        replica = replicas[i]

        return replica


class FaasSystem:
    """

    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.functions = dict()
        self.replicas = defaultdict(list)

        self.request_queue = simpy.Store(env)
        self.scheduler_queue = simpy.Store(env)

        self.load_balancer = RoundRobinLoadBalancer(env, self.replicas)

    def get_replicas(self, fn_name: str, state=None) -> List[FunctionReplica]:
        if state is None:
            return self.replicas[fn_name]

        return [replica for replica in self.replicas[fn_name] if replica.state == state]

    def deploy(self, fn: FunctionDefinition):
        if fn.name in self.functions:
            raise ValueError('function already deployed')

        self.functions[fn.name] = fn

        logger.info('deploying function %s with scale_min=%d', fn.name, fn.scale_min)

        for _ in range(fn.scale_min):
            yield from self.deploy_replica(fn)

    def deploy_replica(self, fn: FunctionDefinition):
        replica = self.create_replica(fn)
        self.replicas[fn.name].append(replica)
        yield self.scheduler_queue.put(replica)

    def invoke(self, request: FunctionRequest):
        # TODO: how to return a FunctionResponse?
        logger.debug('invoking function %s', request.name)

        if request.name not in self.functions:
            logger.warning('invoking non-existing function %s', request.name)
            return

        t_received = self.env.now

        replicas = self.get_replicas(request.name, FunctionState.RUNNING)
        if not replicas:
            '''
            https://docs.openfaas.com/architecture/autoscaling/#scaling-up-from-zero-replicas

            When scale_from_zero is enabled a cache is maintained in memory indicating the readiness of each function.
            If when a request is received a function is not ready, then the HTTP connection is blocked, the function is
            scaled to min replicas, and as soon as a replica is available the request is proxied through as per normal.
            You will see this process taking place in the logs of the gateway component.
            '''
            yield from self.poll_available_replica(request.name)

        if len(replicas) < 1:
            raise ValueError
        elif len(replicas) > 1:
            logger.debug('asking load balancer for replica for request %s:%d', request.name, request.request_id)
            replica = self.next_replica(request)
        else:
            replica = replicas[0]

        logger.debug('dispatching request %s:%d to %s', request.name, request.request_id, replica.node.name)

        yield from simulate_function_invocation(self.env, replica, request)

        # TODO: log trace

    def remove(self):
        # TODO remove deployed function
        raise NotImplementedError

    def next_replica(self, request) -> FunctionReplica:
        return self.load_balancer.next_replica(request)

    def start(self):
        self.env.process(self.run_scheduler_worker())

    def poll_available_replica(self, fn: str, interval=0.5):
        while not self.get_replicas(fn, FunctionState.RUNNING):
            yield self.env.timeout(interval)

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

            replica.node = self.env.get_node_state(result.suggested_host.name)

            # start a new process to simulate starting of pod
            env.process(simulate_function_start(env, replica))

    def create_pod(self, fn: FunctionDefinition):
        return create_function_pod(fn)

    def create_replica(self, fn: FunctionDefinition) -> FunctionReplica:
        replica = FunctionReplica()
        replica.function = fn
        replica.pod = self.create_pod(fn)
        replica.simulator = self.env.simulator_factory.create(self.env, fn)
        return replica


def simulate_function_start(env: Environment, replica: FunctionReplica):
    sim: FunctionSimulator = replica.simulator

    logger.debug('deploying function %s to %s', replica.function.name, replica.node.name)
    yield from sim.deploy(env, replica)
    replica.state = FunctionState.STARTING
    logger.debug('starting function %s on %s', replica.function.name, replica.node.name)
    yield from sim.startup(env, replica)

    logger.debug('running function setup %s on %s', replica.function.name, replica.node.name)
    replica.state = FunctionState.RUNNING
    yield from sim.setup(env, replica)  # FIXME: this is really domain-specific startup


def simulate_function_invocation(env: Environment, replica: FunctionReplica, request: FunctionRequest):
    node = replica.node

    node.current_requests.add(request)
    yield from replica.simulator.invoke(env, replica, request)
    node.current_requests.remove(request)

    # TODO: log traces


class FunctionSimulator(abc.ABC):

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


class SimulatorFactory:

    def create(self, env: Environment, fn: FunctionDefinition) -> FunctionSimulator:
        raise NotImplementedError
