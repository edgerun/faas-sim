import enum
import logging
import math
import time
from collections import defaultdict
from typing import List, Dict, NamedTuple

import simpy

import sim.oracle.oracle as oracles
from core.clustercontext import ClusterContext
from core.model import Pod, Node, SchedulingResult
from core.scheduler import Scheduler
from core.utils import counter, normalize_image_name, parse_size_string
from sim.logging import SimulatedClock, NullLogger, RuntimeLogger
from sim.net import Topology, Flow
from sim.stats import RandomSampler, BufferedSampler

logger = logging.getLogger(__name__)

empty = {}


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

    def __init__(self, function, node: Node = None) -> None:
        super().__init__()
        self.function = function
        self.node = node
        self.state = FunctionState.CONCEIVED

    def __str__(self) -> str:
        return "FunctionReplica(%s, %s, %s)" % (self.function.name, self.node.name if self.node else 'None', self.state)

    def __repr__(self):
        return self.__str__()


class Function:
    """
    A function is a piece of code that runs in a pod.
    """
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
        self.pod: Pod = pod
        self.triggers = triggers

        self.state: FunctionState = FunctionState.CONCEIVED
        self.replicas = []

    def get_resource_requirements(self):
        resource_reqs = [container.resources.requests for container in self.pod.spec.containers]

        total = defaultdict(int)
        for resource_req in resource_reqs:
            for resource, value in resource_req.items():
                total[resource] += value

        return total

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


class FunctionResponse(NamedTuple):
    request_id: int
    code: int
    t_wait: float = 0
    t_exec: float = 0
    node: str = None


class Metrics:
    """
    Instrumentation and trace logger.
    """
    invocations: Dict[str, int]
    last_invocation: Dict[str, float]
    utilization: Dict[str, Dict[str, float]]

    def __init__(self, env, log: RuntimeLogger = None) -> None:
        super().__init__()
        self.env: FaasSimEnvironment = env
        self.logger: RuntimeLogger = log or NullLogger()
        self.invocations = defaultdict(int)
        self.last_invocation = defaultdict(int)
        self.utilization = defaultdict(lambda: defaultdict(float))

    def log(self, metric, value, **tags):
        return self.logger.log(metric, value, **tags)

    def log_network(self, num_bytes, data_type, link):
        tags = dict(link.tags)
        tags['data_type'] = data_type

        self.env.metrics.log('network', num_bytes, **tags)

    def log_scaling(self, function_name, replicas):
        self.env.metrics.log('scale', replicas, function_name=function_name)

    def log_invocation(self, function_name, node_name, t_wait, t_exec):
        self.invocations[function_name] += 1
        self.last_invocation[function_name] = self.env.now

        self.env.metrics.log('invocations', {'t_wait': t_wait, 't_exec': t_exec},
                             function_name=function_name, node=node_name)

    def log_start_exec(self, request: FunctionRequest, replica: FunctionReplica):
        node = replica.node
        function = replica.function

        for resource, value in function.get_resource_requirements().items():
            self.utilization[node.name][resource] += value

        self.env.metrics.log('utilization', {
            'cpu': self.utilization[node.name]['cpu'] / node.capacity.cpu_millis,
            'mem': self.utilization[node.name]['memory'] / node.capacity.memory
        }, node=node.name)

    def log_stop_exec(self, request: FunctionRequest, replica: FunctionReplica):
        node = replica.node
        function = replica.function

        for resource, value in function.get_resource_requirements().items():
            self.utilization[node.name][resource] -= value

        self.env.metrics.log('utilization', {
            'cpu': self.utilization[node.name]['cpu'] / node.capacity.cpu_millis,
            'mem': self.utilization[node.name]['memory'] / node.capacity.memory
        }, node=node.name)

    def get(self, name, **tags):
        return self.logger.get(name, **tags)

    @property
    def clock(self):
        return self.clock

    @property
    def records(self):
        return self.logger.records


class FaasSimEnvironment(simpy.Environment):

    def __init__(self, topology: Topology, cluster_context: ClusterContext = None, initial_time=0):
        super().__init__(initial_time)

        self.request_generator = object
        self.request_queue = simpy.Store(self)
        self.scheduler_queue = simpy.Store(self)
        self.topology: Topology = topology
        topology.create_index()

        # allows us to inject a pre-calculated bandwidth graph that was cached
        if cluster_context is None:
            self.cluster: ClusterContext = topology.create_cluster_context()
        else:
            self.cluster: ClusterContext = cluster_context

        self.scheduler = Scheduler(self.cluster)
        self.faas_gateway = FaasGateway(self)
        self.execution_simulator = ExecutionSimulator(self)

        self.clock = SimulatedClock(self)
        self.metrics = Metrics(self, RuntimeLogger(self.clock))

        self.execution_time_oracle = oracles.FittedExecutionTimeOracle()
        self.startup_time_oracle = oracles.HackedFittedStartupTimeOracle()


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
    }, node=node.name)

    if func.state != FunctionState.RUNNING:
        # synchronize function state
        func.state = FunctionState.STARTING

    then = env.now
    # simulate docker pull
    if result.needed_images:
        yield from simulate_docker_pull(env, replica, result)

    # simulate container startup (we use the hacked version which estimates only pure startup)
    _, t = env.startup_time_oracle.estimate(env.cluster, func.pod, result)
    t = float(t)

    logger.debug('function start: (%s, %s, %.4f)', func.name, node.name, t)

    yield env.timeout(t)  # simulate startup (image download (perhaps) + container startup + program startup)

    env.metrics.log('startup', env.now - then, function_name=func.name, node=node, images=len(result.needed_images))
    env.metrics.log('replicas', len(func.replicas), function_name=func.name)
    replica.state = FunctionState.RUNNING
    func.state = FunctionState.RUNNING
    env.faas_gateway.replicas.free(replica)


def simulate_docker_pull(env: FaasSimEnvironment, replica: FunctionReplica, result: SchedulingResult):
    # TODO: there's a lot of potential to improve fidelity here: consider image layers, simulate extraction time, etc.
    node = result.suggested_host

    sizes = env.cluster.get_image_sizes(replica.function.pod, node.labels['beta.kubernetes.io/arch'])
    # needed image names are already normalized by the scheduler
    required = sum([size for image, size in sizes.items() if normalize_image_name(image) in result.needed_images])

    if required <= 0:
        return

    route = env.topology.get_route(env.topology.get_registry(), node)
    flow = Flow(env, required, route)
    yield flow.start()
    for hop in route.hops:
        env.metrics.log_network(required, 'docker_pull', hop)


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
            # TODO: reconcile with ReplicaPool
            return simpy.Resource(env, capacity=1)

        self.resources: Dict[FunctionReplica, simpy.Resource] = defaultdict(resource_factory)

    def run(self, req: FunctionRequest, replica: FunctionReplica):
        """
        Is expected to run in a separate process.
        """
        env = self.env
        node = replica.node

        func = env.faas_gateway.functions[req.name]

        yield from self.simulate_data_download(replica)

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
            env.metrics.log_start_exec(req, replica)

            wait = env.now - arrive
            logger.debug('%.2f function request %s waited %.2f', env.now, req, wait)

            yield env.timeout(t)

        yield from self.simulate_data_upload(replica)

        self.running[replica].remove(req)
        env.metrics.log_stop_exec(req, replica)

        if func.triggers:  # simulates function compositions
            for next_func in func.triggers:
                # example: in an ML workflow, a pre-processing step may after its completion trigger a training step
                env.request_queue.put(FunctionRequest(next_func, empty))

    def simulate_data_download(self, replica: FunctionReplica):
        node = replica.node
        func = replica.function
        env = self.env

        if 'data.skippy.io/receives-from-storage' not in func.pod.spec.labels:
            return

        size = parse_size_string(func.pod.spec.labels['data.skippy.io/receives-from-storage'])

        storage_node_name = env.cluster.get_next_storage_node(node)  # FIXME

        if storage_node_name == node.name:
            # FIXME this is essentially a disk read and not a network connection
            yield env.timeout(size / 1.25e+8)  # 1.25e+8 = 1 GBit/s
            return

        storage_node = env.cluster.get_node(storage_node_name)
        route = env.topology.get_route(storage_node, node)
        flow = Flow(env, size, route)
        yield flow.start()
        for hop in route.hops:
            env.metrics.log_network(size, 'data_download', hop)

    def simulate_data_upload(self, replica: FunctionReplica):
        node = replica.node
        func = replica.function
        env = self.env

        if 'data.skippy.io/sends-to-storage' not in func.pod.spec.labels:
            return

        size = parse_size_string(func.pod.spec.labels['data.skippy.io/sends-to-storage'])

        storage_node_name = env.cluster.get_next_storage_node(node)  # FIXME

        if storage_node_name == node.name:
            # FIXME this is essentially a disk read and not a network connection
            yield env.timeout(size / 1.25e+8)  # 1.25e+8 = 1 GBit/s
            return

        storage_node = env.cluster.get_node(storage_node_name)
        route = env.topology.get_route(node, storage_node)
        flow = Flow(env, size, route)
        yield flow.start()
        for hop in route.hops:
            env.metrics.log_network(size, 'data_upload', hop)


class ReplicaPool:
    """
    This currently does not accurately simulate round-robin load balancing. It's probably more like a least-requested.
    After a replica is done serving a request, it is immediately returned to the pool. This is not how OpenFaaS would
    work with its watchdog and synchronous request execution.
    """
    stores: Dict[str, simpy.Store]

    def __init__(self, env: FaasSimEnvironment) -> None:
        super().__init__()
        self.env = env
        self.stores = defaultdict(lambda: simpy.Store(env))

    def free(self, replica: FunctionReplica):
        store = self.stores[replica.function.name]
        return store.put(replica)

    def request(self, function: Function):
        store = self.stores[function.name]
        return store.get()


class FaasGateway:

    def __init__(self, env: FaasSimEnvironment) -> None:
        super().__init__()
        self.env = env
        self.functions: Dict[str, Function] = dict()
        self.replicas: ReplicaPool = ReplicaPool(env)

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
        if function_name not in self.functions:
            raise ValueError

        function = self.functions[function_name]
        num_replicas = len(function.replicas)

        if num_replicas <= function.scale_min:
            # already at min replicas
            return 0

        replicas_to_remove = num_replicas - function.scale_min

        # TODO: check with OpenFaaS how replicas to scale down are selected
        replicas = [function.replicas[i] for i in range(replicas_to_remove)]

        for replica in replicas:
            self._remove_replica(replica)

        return replicas_to_remove

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
            self._remove_replica(replica)

        function.state = FunctionState.SUSPENDED

    def wakeup(self, function: str):
        if function not in self.functions:
            raise ValueError

        func = self.functions[function]

        if func.state != FunctionState.SUSPENDED:
            return

        func.state = FunctionState.STARTING
        self._schedule_replicas(func, func.scale_min)

    def request(self, request: FunctionRequest) -> FunctionResponse:
        if request.name not in self.functions:
            return FunctionResponse(request.request_id, 404)

        env = self.env
        t_received = env.now
        func = self.functions[request.name]

        if func.state == FunctionState.CONCEIVED:
            return FunctionResponse(request.request_id, 404)
        elif func.state == FunctionState.SUSPENDED:
            '''
            https://docs.openfaas.com/architecture/autoscaling/#scaling-up-from-zero-replicas

            When scale_from_zero is enabled a cache is maintained in memory indicating the readiness of each function.
            If when a request is received a function is not ready, then the HTTP connection is blocked, the function is
            scaled to min replicas, and as soon as a replica is available the request is proxied through as per normal.
            You will see this process taking place in the logs of the gateway component.
            '''
            self.wakeup(func.name)
            # TODO: wait *explicitly* for replica to become available (instead of implicitly through the ReplicaPool)

        while True:
            replica = yield self.replicas.request(func)  # TODO: timeout
            if replica.state == FunctionState.RUNNING:
                # there may still be replicas in the pool which have been suspended, i.e., removed
                break

        try:
            t_started = env.now
            yield from env.execution_simulator.run(request, replica)
            t_done = env.now
            t_wait = t_started - t_received
            t_exec = t_done - t_started
            response = FunctionResponse(request.request_id, 200, t_wait, t_exec, replica.node.name)
            env.metrics.log_invocation(func.name, replica.node.name, t_wait, t_exec)
            return response
        finally:
            self.replicas.free(replica)

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

    def request_worker(self):
        env = self.env
        gateway = self

        while True:
            req: FunctionRequest
            req = yield env.request_queue.get()
            env.process(gateway.request(req))

    def faas_idler(self, inactivity_duration=300, reconcile_interval=30):
        """
        https://github.com/openfaas-incubator/faas-idler
        https://github.com/openfaas-incubator/faas-idler/blob/master/main.go

        default values:
        https://github.com/openfaas-incubator/faas-idler/blob/668991c532156275993399ee79a297a4c2d651ec/docker-compose.yml

        :param inactivity_duration: i.e. 15m (Golang duration)
        :param reconcile_interval: i.e. 1m (default value)
        :return: an event generator
        """

        while True:
            yield self.env.timeout(reconcile_interval)

            for function in self.functions.values():
                if not function.scale_zero:
                    continue
                if function.state != FunctionState.RUNNING:
                    continue

                idle_time = self.env.now - self.env.metrics.last_invocation[function.name]
                if idle_time >= inactivity_duration:
                    self.suspend(function.name)
                    logger.debug('function %s has been idle for %.2fs', function.name, idle_time)

    def _schedule_replicas(self, function, num):
        logger.debug('scheduling %d replicas for %s', num, function)
        env = self.env

        for i in range(num):
            replica = FunctionReplica(function)
            function.replicas.append(replica)
            env.scheduler_queue.put(replica)

        env.metrics.log_scaling(function.name, len(function.replicas))

    def _remove_replica(self, replica):
        env = self.env
        node = replica.node

        self.env.cluster.remove_pod_from_node(replica.function.pod, node)
        replica.state = FunctionState.SUSPENDED
        replica.function.replicas.remove(replica)

        env.metrics.log('allocation', {
            'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
            'mem': 1 - (node.allocatable.memory / node.capacity.memory)
        }, node=node.name)
        env.metrics.log_scaling(replica.function.name, len(replica.function.replicas))
