import enum
import logging
import math
import time
from typing import List, NamedTuple, Dict

import simpy

import sim.oracle.oracle as oracles
import sim.synth.pods as pods
from core.clustercontext import ClusterContext
from core.model import Pod, Node, SchedulingResult
from core.scheduler import Scheduler
from sim.logging import SimulatedClock, NullLogger, PrintLogger
from sim.simclustercontext import SimulationClusterContext
from sim.stats import RandomSampler, ParameterizedDistribution, BufferedSampler
from sim.synth.bandwidth import generate_bandwidth_graph
from sim.synth.nodes import node_factory_cloud_majority, node_synthesizer

logger = logging.getLogger(__name__)

empty = {}


class FunctionState(enum.Enum):
    CONCEIVED = 1
    STARTING = 2
    RUNNING = 3
    SUSPENDED = 4


class FunctionReplica:
    function: 'Function'
    node: Node
    state: FunctionState

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
        self.node = None
        self.state = FunctionState.CONCEIVED


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


FunctionRequest = NamedTuple('FunctionRequest', [('name', str), ('body', dict)])


class FaasSimEnvironment(simpy.Environment):

    def __init__(self, cluster: ClusterContext, initial_time=0):
        super().__init__(initial_time)

        self.function_synthesizer = object
        self.request_generator = object
        self.request_queue = simpy.Store(self)
        self.scheduler_queue = simpy.Store(self)
        self.cluster: ClusterContext = cluster
        self.scheduler = Scheduler(self.cluster)
        self.faas_gateway = FaasGateway(self)

        self.clock = SimulatedClock(self)
        self.metrics = PrintLogger(self.clock)
        # self.metrics = NullLogger(self.clock)

        self.execution_time_oracle = oracles.FittedExecutionTimeOracle()
        self.startup_time_oracle = oracles.FittedStartupTimeOracle()

        self.functions = {
            'preprocess': Function('preprocess', pods.create_ml_wf_1_pod(1), ['train']),
            'train': Function('train', pods.create_ml_wf_2_pod(2)),
            'inference': Function('inference', pods.create_ml_wf_3_serve(3))
        }


def request_generator(env: FaasSimEnvironment, arrival_profile: RandomSampler, request_factory):
    sampler = BufferedSampler(arrival_profile)

    while True:
        ia = sampler.sample()
        logger.debug('next request: %.4f', ia)
        env.request_queue.put(request_factory())
        yield env.timeout(ia)


def dispatch_call(env: FaasSimEnvironment, req: FunctionRequest, nodes: List[Node]):
    # TODO: there would be load balancing here, but we assume max replicas of 1
    node = nodes[0]
    logger.debug('dispatching req to function %s to node %s', req.name, node.name)

    yield from simulate_execution(env, req, node)


def simulate_startup(env: FaasSimEnvironment, replica: FunctionReplica, result: SchedulingResult):
    replica.state = FunctionState.STARTING
    func = replica.function

    if func.state != FunctionState.RUNNING:
        # synchronize function state
        func.state = FunctionState.STARTING

    _, t = env.startup_time_oracle.estimate(env.cluster, func.pod, result)
    t = float(t)

    logger.debug('function start: (%s, %s, %.4f)', func.name, result.suggested_host.name, t)

    yield env.timeout(t)  # simulate startup (image download (perhaps) + container startup + program startup)

    env.metrics.log('startup', t, function_name=func.name, node=result.suggested_host)
    env.metrics.log('replicas', len(func.replicas), function_name=func.name)
    replica.state = FunctionState.RUNNING
    func.state = FunctionState.RUNNING


def simulate_execution(env: FaasSimEnvironment, req: FunctionRequest, node: Node):
    func = env.functions[req.name]

    _, t = env.execution_time_oracle.estimate(env.cluster, func.pod, SchedulingResult(node, 1, []))
    t = float(t)

    logger.debug('function execution: (%s, %s, %.4f)', req.name, node.name, t)

    yield env.timeout(t)

    env.metrics.log('invocations', t, function_name=func.name, node=node.name)

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
        return [replica.node for replica in function.replicas if replica.state == FunctionState.RUNNING]

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

        nodes = self.discover(function)

        for node in nodes:
            self.env.cluster.remove_pod_from_node(function.pod, node)

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
                nodes = gateway.discover(func)

                if not nodes:
                    raise ValueError('state error: func is deployed but no nodes were discovered')

                env.process(dispatch_call(env, req, nodes))

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


class Simulation:

    def __init__(self, cluster) -> None:
        super().__init__()
        self.cluster = cluster
        self.env = FaasSimEnvironment(self.cluster)
        self.env.process(self.env.faas_gateway.request_worker())
        self.env.process(self.env.faas_gateway.scheduler_worker())

        for function in self.env.functions.values():
            self.env.faas_gateway.deploy(function)

        training_trigger = request_generator(
            self.env,
            ParameterizedDistribution.expon(((300, 300,), None, None)),
            lambda: FunctionRequest('preprocess', empty)
        )
        self.env.process(training_trigger)

        inference_trigger = request_generator(
            self.env,
            ParameterizedDistribution.expon(((25, 50), None, None)),
            lambda: FunctionRequest('inference', empty)
        )
        self.env.process(inference_trigger)

    def run(self, until):
        env = self.env

        env.run(until=until)
        print('simulation time is now: %.2f' % env.now)


def main():
    logging.basicConfig(level=logging.DEBUG)

    oracles.data_dir = '/home/thomas/workspace/serverless-edge-ai/sched-sim/sim/oracle/data'

    # TODO: inject topology
    gen = node_synthesizer(node_factory_cloud_majority)
    nodes = [next(gen) for i in range(20)]
    topology = generate_bandwidth_graph(nodes)
    cluster = SimulationClusterContext(nodes, topology)

    sim = Simulation(cluster)

    then = time.time()
    sim.run(60 * 60 * 24)
    print('simulation took %.2f ms' % ((time.time() - then) * 1000))


if __name__ == '__main__':
    main()
