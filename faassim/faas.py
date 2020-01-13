import enum
import logging
import math
import time
from collections import defaultdict
from typing import List, Dict, Tuple

import simpy

import sim.oracle.oracle as oracles
from core.clustercontext import ClusterContext
from core.model import Pod, Node, SchedulingResult
from core.scheduler import Scheduler
from sim.logging import SimulatedClock, NullLogger
from sim.stats import RandomSampler, BufferedSampler

logger = logging.getLogger(__name__)

empty = {}


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


class FunctionReplica:
    """
    Represents an instance of a pod serving a function on a node.
    """

    function: 'Function'
    node: Node
    state: FunctionState

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
        self.node = None
        self.state = FunctionState.CONCEIVED

    def __str__(self) -> str:
        return "FunctionReplica(%s, %s, %s)" % (self.function.name, self.node.name if self.node else 'None', self.state)

    def __repr__(self):
        return self.__str__()


class Function:
    name: str
    pod: Pod
    state: FunctionState
    replicas: List[FunctionReplica]
    triggers: List[str]

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 20
    scale_zero: bool = False

    def __init__(self, name, pod, triggers: List[str] = None) -> None:
        super().__init__()
        self.name = name
        self.pod = pod
        self.triggers = triggers

        self.state = FunctionState.CONCEIVED
        self.replicas = []

    def __str__(self) -> str:
        return "Function%s" % self.__dict__


class FunctionRequest:
    request_id: int
    name: str
    body: str

    id_generator = counter()

    def __init__(self, name, body=None) -> None:
        super().__init__()
        self.name = name
        self.body = body or empty
        self.request_id = next(self.id_generator)

    def __str__(self) -> str:
        return 'FunctionRequest(%d, %s, %s)' % (self.request_id, self.name, self.body)

    def __repr__(self):
        return self.__str__()


class FaasSimEnvironment(simpy.Environment):

    def __init__(self, cluster: ClusterContext, initial_time=0):
        super().__init__(initial_time)

        self.request_generator = object
        self.request_queue = simpy.Store(self)
        self.scheduler_queue = simpy.Store(self)
        self.cluster: ClusterContext = cluster
        self.scheduler = Scheduler(self.cluster)
        self.faas_gateway = FaasGateway(self)
        self.execution_simulator = ExecutionSimulator(self)

        self.clock = SimulatedClock(self)
        # self.metrics = PrintLogger(self.clock)
        self.metrics = NullLogger(self.clock)

        self.execution_time_oracle = oracles.FittedExecutionTimeOracle()
        self.startup_time_oracle = oracles.FittedStartupTimeOracle()


def request_generator(env: FaasSimEnvironment, arrival_profile: RandomSampler, request_factory):
    sampler = BufferedSampler(arrival_profile)

    while True:
        ia = sampler.sample()
        logger.debug('next request: %.4f', ia)
        env.request_queue.put(request_factory())
        yield env.timeout(ia)


def dispatch_call(env: FaasSimEnvironment, req: FunctionRequest, replicas: List[FunctionReplica]):
    # TODO: there would be load balancing here, but we assume max replicas of 1
    replica = replicas[0]
    logger.debug('dispatching req to function %s to node %s', req.name, replica.node.name)

    yield from env.execution_simulator.run(req, replica)


def simulate_startup(env: FaasSimEnvironment, replica: FunctionReplica, result: SchedulingResult):
    replica.state = FunctionState.STARTING
    func = replica.function
    node = result.suggested_host

    env.metrics.log('allocation', {
        'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
        'mem': 1 - (node.allocatable.memory / node.capacity.memory)
    }, node=node)

    if func.state != FunctionState.RUNNING:
        # synchronize function state
        func.state = FunctionState.STARTING

    _, t = env.startup_time_oracle.estimate(env.cluster, func.pod, result)
    t = float(t)

    logger.debug('function start: (%s, %s, %.4f)', func.name, node.name, t)

    yield env.timeout(t)  # simulate startup (image download (perhaps) + container startup + program startup)

    env.metrics.log('startup', t, function_name=func.name, node=node)
    env.metrics.log('replicas', len(func.replicas), function_name=func.name)
    replica.state = FunctionState.RUNNING
    func.state = FunctionState.RUNNING


class ExecutionSimulator:
    """
    Each FunctionReplica has a max concurrency level of 1. Every FunctionReplica represents a pod that hosts a function.
    """

    def __init__(self, env: FaasSimEnvironment) -> None:
        super().__init__()
        self.env = env
        # keeps track of currently running functions on a node, to enforce concurrency limits
        self.running: Dict[FunctionReplica, List[FunctionRequest]] = defaultdict(list)

        def resource_factory():
            # each function replica can serve one request at a time (concurrency limit = 1)
            return simpy.Resource(env, capacity=1)

        self.resources: Dict[FunctionReplica, simpy.Resource] = defaultdict(resource_factory)
        self.utilization_cpu: Dict[Node, float] = defaultdict(0)
        self.utilization_mem: Dict[Node, float] = defaultdict(0)

    def run(self, req: FunctionRequest, replica: FunctionReplica):
        """
        Is expected to run in a separate process.
        """
        env = self.env
        node = replica.node

        func = env.faas_gateway.functions[req.name]

        # estimate execution time
        _, t = env.execution_time_oracle.estimate(env.cluster, func.pod, SchedulingResult(node, 1, []))
        t = float(t)

        logger.debug('function execution: (%s, %s, %.4f)', req.name, node.name, t)
        self.running[replica].append(req)
        logger.debug('currently running functions: %s', self.running)

        resource = self.resources[replica]

        arrive = env.now
        logger.debug('%.2f function request %s arrived', arrive, req)
        with resource.request() as lock:
            yield lock  # waits for lock acquisition, i.e., for any previous function execution to finish

            wait = env.now - arrive
            logger.debug('%.2f function request %s waited %.2f', env.now, req, wait)

            yield env.timeout(t)

            self.running[replica].remove(req)
            env.metrics.log('invocations', {'t_wait': wait, 't_exec': t}, function_name=func.name, node=node.name)

            if func.triggers:  # simulates function compositions
                for next_func in func.triggers:
                    # example: in an ML workflow, a pre-processing step may after its completion trigger a training step
                    env.request_queue.put(FunctionRequest(next_func, empty))


class FaasGateway:

    def __init__(self, env: FaasSimEnvironment) -> None:
        super().__init__()
        self.env = env
        self.functions: Dict[str, Function] = dict()

    def discover(self, function: Function):
        return [replica for replica in function.replicas if replica.state == FunctionState.RUNNING]

    def deploy(self, function: Function):
        # FIXME: blocks replication, which is fine because we're currently not simulating it
        # if function.name in self.functions:
        #     return

        logger.debug('deploying function %s', function.name)

        # deploy means registering the function and creating the number of min replicas
        self.functions[function.name] = function
        self._schedule_replicas(function, function.scale_min)

    def scale_down(self, function_name: str):
        raise NotImplementedError  # TODO

    def scale_up(self, function_name: str):
        if function_name not in self.functions:
            raise ValueError

        function = self.functions[function_name]
        num_replicas = len(function.replicas)

        if num_replicas >= function.scale_max:
            # already at max replicas
            return 0

        if function.scale_factor == 0:
            return 0

        max_replicas_to_add = function.scale_max - num_replicas
        factor = min(max(function.scale_factor, 1), 100) / 100
        replicas_to_add = min(max(math.ceil(function.scale_max * factor), 1), max_replicas_to_add)

        if replicas_to_add <= 0:
            # shouldn't happen
            raise ValueError

        self._schedule_replicas(function, replicas_to_add)
        return replicas_to_add

    def suspend(self, function_name: str):
        if function_name not in self.functions:
            raise ValueError

        function = self.functions[function_name]

        replicas = self.discover(function)

        for replica in replicas:
            self.env.cluster.remove_pod_from_node(function.pod, replica.node)

    def wakeup(self, function: str):
        if function not in self.functions:
            raise ValueError

    def request(self, request: FunctionRequest):
        if request.name not in self.functions:
            raise ValueError  # maybe allow 404 requests?

        '''
        https://docs.openfaas.com/architecture/autoscaling/#scaling-up-from-zero-replicas
        
        If when a request is received a function is not ready, then the HTTP connection is blocked, the function is
        scaled to min replicas, and as soon as a replica is available the request is proxied through as per normal. You
        will see this process taking place in the logs of the gateway component.
        '''

        # TODO
        pass

    def _schedule_replicas(self, function, num):
        logger.debug('scheduling %d replicas for %s', num, function)

        for i in range(num):
            replica = FunctionReplica(function)
            function.replicas.append(replica)
            self.env.scheduler_queue.put(replica)

    def scheduler_worker(self):
        env = self.env
        gateway = env.faas_gateway

        while True:
            replica: FunctionReplica
            replica = yield env.scheduler_queue.get()

            func = replica.function

            # schedule the required pod
            pod = gateway.functions[func.name].pod
            then = time.time()
            result = env.scheduler.schedule(pod)
            duration = time.time() - then
            yield env.timeout(duration)  # include scheduling latency in simulation time
            logger.debug('Pod scheduling took %.2f ms, and yielded %s', duration * 1000, result)

            if not result.suggested_host:
                raise RuntimeError('pod %s cannot be scheduled' % pod.name)

            replica.node = result.suggested_host

            # start a new process to simulate starting of pod
            env.process(simulate_startup(env, replica, result))
            # TODO

    def request_worker(self):
        """
        The main FaaS control loop, which dispatches function requests from a queue to running pods, or informs the
        scheduler if necessary. In OpenFaaS, the API Gateway is the entry point through which every incoming call passes,
        and which talks to faas-netes in the case of using the Kubernetes runtime.

        TODO: ideally, replicate the high-level API of faas-provider https://github.com/openfaas/faas-provider/
        """
        env = self.env
        gateway = self

        while True:
            req: FunctionRequest
            req = yield env.request_queue.get()

            try:
                func: Function = gateway.functions[req.name]
            except KeyError:
                logger.debug('requested a function %s but was not deployed', req.name)
                continue

            if func.state == FunctionState.STARTING:
                # TODO: needs some form of buffering
                logger.warning('discarding function request %s', func)
                continue

            elif func.state == FunctionState.RUNNING:
                # if a function pod is running on a node, route the request there
                replicas = gateway.discover(func)

                if not replicas:
                    raise ValueError('state error: func is deployed but no nodes were discovered')

                env.process(dispatch_call(env, req, replicas))

            elif func.state == FunctionState.SUSPENDED:
                # if not, the function was scaled to zero, and we need to scale up and defer the request
                raise NotImplementedError

            elif func.state == FunctionState.CONCEIVED:
                logger.warning('discarding function request %s', func)
                continue
            else:
                raise ValueError('Unhandled state %s', func.state)

    def faas_idler(self):
        # TODO
        pass
