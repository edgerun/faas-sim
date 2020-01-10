import enum
import logging
import time
from typing import List, NamedTuple

import simpy

import sim.synth.pods as pods
from core.clustercontext import ClusterContext
from core.model import Pod, Node
from core.scheduler import Scheduler
from sim.simclustercontext import SimulationClusterContext
from sim.synth.bandwidth import generate_bandwidth_graph
from sim.synth.nodes import create_cloud_node

log = logging.getLogger(__name__)

empty = {}


class FunctionState(enum.Enum):
    CONCEIVED = 1
    STARTING = 2
    RUNNING = 3
    SUSPENDED = 4


class Function:
    name: str
    pod: Pod
    state: FunctionState
    replicas: int
    triggers: list

    def __init__(self, name, pod, triggers: List[str] = None) -> None:
        super().__init__()
        self.name = name
        self.pod = pod
        self.triggers = triggers

        self.state = FunctionState.CONCEIVED
        self.replicas = 0

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
        self.oracles = []

        self.functions = {
            'ml_0_wf_1': Function('ml_0_wf_1', pods.create_ml_wf_1_pod(1), ['ml_0_wf_2']),
            'ml_0_wf_2': Function('ml_0_wf_2', pods.create_ml_wf_1_pod(2)),
            'ml_0_wf_3': Function('ml_0_wf_3', pods.create_ml_wf_1_pod(2))
        }


def request_generator(env: FaasSimEnvironment):
    while True:
        env.request_queue.put(FunctionRequest('ml_0_wf_1', empty))
        yield env.timeout(1)


def dispatch_call(env: FaasSimEnvironment, req: FunctionRequest, nodes: List[Node]):
    # TODO: there would be load balancing here, but we assume max replicas of 1
    node = nodes[0]
    log.debug('dispatching req to function %s to node %s', req.name, node.name)

    yield from simulate_execution(env, req, node, cold=False)


def simulate_execution(env: FaasSimEnvironment, req: FunctionRequest, node: Node, cold=False):
    log.debug('simulating the execution of %s on %s', req.name, node.name)
    func = env.functions[req.name]

    if cold:
        log.debug('function %s has a cold start', req.name)
        # TODO: check if image is available, and simulate pulling of image if not
        yield env.timeout(1)
        # TODO: and starting of container
        yield env.timeout(1)

    # TODO: simulate execution
    log.debug('simulating execution of %s ...', req.name)
    yield env.timeout(1)

    if func.triggers:
        for next_func in func.triggers:
            # example: in an ML workflow, a pre-processing step may after its completion trigger a training step
            env.request_queue.put(FunctionRequest(next_func, empty))

    # TODO: when to free resources?
    pass


class FaasGateway:

    def __init__(self, env: FaasSimEnvironment) -> None:
        super().__init__()
        self.env = env
        self.functions = dict()

    def discover(self, function: Function):
        # this is basically the service discovery step, which would typically done via a table. could speed this up.
        return [node for node in self.env.cluster.list_nodes() if function.pod in node.pods]

    def deploy(self, function: Function):
        if function.name in self.functions:
            return

        self.functions[function.name] = function
        self.env.scheduler_queue.put(function.pod)

    def scale(self, function: str, replicas: int):
        if function not in self.functions:
            raise ValueError

        # TODO

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


def faas_scheduler_worker(env):
    gateway = env.faas_gateway

    while True:
        func: Function
        func = yield env.scheduler_queue.get()

        # schedule the required pod
        pod = gateway.functions[func.name].pod
        then = time.time()
        result = env.scheduler.schedule(pod)
        duration = time.time() - then
        yield env.timeout(duration)
        logging.debug('Pod scheduling took %.2f ms, and yielded %s', duration * 1000, result)

        if not result.suggested_host:
            raise RuntimeError('pod %s cannot be scheduled' % pod.name)

        # start a new process to simulate starting of pod
        # TODO


def faas_request_worker(env: FaasSimEnvironment):
    """
    The main FaaS control loop, which dispatches function requests from a queue to running pods, or informs the
    scheduler if necessary. In OpenFaaS, the API Gateway is the entry point through which every incoming call passes,
    and which talks to faas-netes in the case of using the Kubernetes runtime.

    TODO: ideally, replicate the high-level API of faas-provider https://github.com/openfaas/faas-provider/
    """
    gateway = env.faas_gateway

    while True:
        req: FunctionRequest
        req = yield env.request_queue.get()

        try:
            func: Function = gateway.functions[req.name]
        except KeyError:
            log.debug('requested a function %s but was not deployed', req.name)
            continue

        if func.state == FunctionState.STARTING:
            # TODO: needs some form of buffering
            raise NotImplementedError

        elif func.state == FunctionState.RUNNING:
            # if a function pod is running on a node, route the request there
            nodes = gateway.discover(func)

            if not nodes:
                raise ValueError('state error: func is deployed but no nodes were discovered')

            env.process(dispatch_call(env, req, nodes))

        elif func.state == FunctionState.SUSPENDED:
            # if not, the function was scaled to zero, and we need to scale up and defer the request
            raise NotImplementedError

        else:
            raise ValueError('Unhandled state %s', func.state)


class Simulation:

    def __init__(self, cluster) -> None:
        super().__init__()
        self.cluster = cluster
        self.env = FaasSimEnvironment(self.cluster)
        self.env.process(faas_request_worker(self.env))
        self.env.process(faas_scheduler_worker(self.env))

        self.env.process(request_generator(self.env))

    def run(self):
        env = self.env

        env.run(until=30)
        print('simulation time is now: %.2f' % env.now)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # TODO: inject topology
    nodes = [
        create_cloud_node(1),
        create_cloud_node(2),
        create_cloud_node(3)
    ]
    topology = generate_bandwidth_graph(nodes)
    cluster = SimulationClusterContext(nodes, topology)

    sim = Simulation(cluster)

    then = time.time()
    sim.run()
    print('simulation took %.2f ms' % ((time.time() - then) * 1000))


if __name__ == '__main__':
    main()
