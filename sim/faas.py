import abc
import enum
import logging
import time
from collections import defaultdict
from typing import List, Dict, NamedTuple

import simpy
from ether.util import parse_size_string
from skippy.core.model import Pod

from sim.core import Environment, NodeState
from sim.skippy import create_function_pod

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
    labels: Dict[str, str]

    requests: Resources = Resources()

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 20
    scale_zero: bool = False

    def __init__(self, name: str, image: str) -> None:
        super().__init__()
        self.name = name
        self.image = image
        self.labels = {}

    def get_resource_requirements(self) -> Dict:
        return {
            'cpu': self.requests.cpu,
            'memory': self.requests.memory
        }


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

        # TODO: fix skippy.get_init_image_states first
        # self.env.metrics.log_function_definition(fn)
        self.env.metrics.log_function_deploy(fn)

        logger.info('deploying function %s with scale_min=%d', fn.name, fn.scale_min)
        self.env.metrics.log_scaling(fn.name, fn.scale_min)

        for _ in range(fn.scale_min):
            yield from self.deploy_replica(fn)

    def deploy_replica(self, fn: FunctionDefinition):
        replica = self.create_replica(fn)
        self.replicas[fn.name].append(replica)
        self.env.metrics.log_queue_schedule(replica)
        self.env.metrics.log_function_replica(replica)
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

        t_start = self.env.now
        yield from simulate_function_invocation(self.env, replica, request)

        t_end = self.env.now

        t_wait = t_start - t_received
        t_exec = t_end - t_start
        self.env.metrics.log_invocation(replica.function.name, replica.node.name, t_wait, t_start, t_exec, id(replica))

    def remove(self):
        # TODO remove deployed function
        # TODO log scaling (removal)
        # TODO log teardown
        raise NotImplementedError

    def next_replica(self, request) -> FunctionReplica:
        return self.load_balancer.next_replica(request)

    def start(self):
        for process in self.env.background_processes:
            self.env.process(process(self.env))
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
            self.env.metrics.log_start_schedule(replica)
            pod = replica.pod
            then = time.time()
            result = env.scheduler.schedule(pod)
            duration = time.time() - then
            self.env.metrics.log_finish_schedule(replica, result)

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

    def discover(self, function: FunctionDefinition) -> List[FunctionReplica]:
        return [replica for replica in self.replicas[function.name] if replica.state == FunctionState.RUNNING]

    def _remove_replica(self, replica: FunctionReplica):
        env = self.env
        node = replica.node.ether_node

        env.metrics.log_teardown(replica)
        yield from replica.simulator.teardown(env, replica)

        self.env.cluster.remove_pod_from_node(replica.pod, node)
        replica.state = FunctionState.SUSPENDED
        self.replicas[replica.function.name].remove(replica)

        env.metrics.log('allocation', {
            'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
            'mem': 1 - (node.allocatable.memory / node.capacity.memory)
        }, node=node.name)
        env.metrics.log_scaling(replica.function.name, -1)

    def suspend(self, function_name: str):
        if function_name not in self.functions:
            raise ValueError

        function: FunctionDefinition = self.functions[function_name]
        replicas: List[FunctionReplica] = self.discover(function)

        for replica in replicas:
            self._remove_replica(replica)

        self.env.metrics.log_function_suspend(function)


def simulate_function_start(env: Environment, replica: FunctionReplica):
    sim: FunctionSimulator = replica.simulator

    logger.debug('deploying function %s to %s', replica.function.name, replica.node.name)
    env.metrics.log_deploy(replica)
    yield from sim.deploy(env, replica)
    replica.state = FunctionState.STARTING
    env.metrics.log_startup(replica)
    logger.debug('starting function %s on %s', replica.function.name, replica.node.name)
    yield from sim.startup(env, replica)

    logger.debug('running function setup %s on %s', replica.function.name, replica.node.name)
    replica.state = FunctionState.RUNNING
    env.metrics.log_setup(replica)
    yield from sim.setup(env, replica)  # FIXME: this is really domain-specific startup
    env.metrics.log_finish_deploy(replica)


def simulate_function_invocation(env: Environment, replica: FunctionReplica, request: FunctionRequest):
    node = replica.node

    node.current_requests.add(request)
    env.metrics.log_start_exec(request, replica)

    yield from replica.simulator.invoke(env, replica, request)

    env.metrics.log_stop_exec(request, replica)
    node.current_requests.remove(request)


def faas_idler(env: Environment, inactivity_duration=300, reconcile_interval=30):
    """
    https://github.com/openfaas-incubator/faas-idler
    https://github.com/openfaas-incubator/faas-idler/blob/master/main.go

    default values:
    https://github.com/openfaas-incubator/faas-idler/blob/668991c532156275993399ee79a297a4c2d651ec/docker-compose.yml

    :param env: the faas environment
    :param inactivity_duration: i.e. 15m (Golang duration)
    :param reconcile_interval: i.e. 1m (default value)
    :return: an event generator
    """
    faas: FaasSystem = env.faas
    while True:
        yield env.timeout(reconcile_interval)

        for function in faas.functions.values():
            if not function.scale_zero:
                continue
            if function.state != FunctionState.RUNNING:
                continue

            idle_time = env.now - env.metrics.last_invocation[function.name]
            if idle_time >= inactivity_duration:
                faas.suspend(function.name)
                logger.debug('%.2f function %s has been idle for %.2fs', env.now, function.name, idle_time)


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
