import logging
from typing import Optional, Callable

from ether.blocks.nodes import create_node
from ether.cell import LANCell
from simpy import Resource
from skippy.core.model import SchedulingResult
from skippy.core.utils import parse_size_string

from faassim.oracle.oracle import FittedExecutionTimeOracle
from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionDefinition, FunctionReplica, FunctionSimulator, SimulatorFactory, FunctionRequest
from sim.faassim import Simulation
from sim.logging import RuntimeLogger, SimulatedClock
from sim.metrics import Metrics
from sim.topology import Topology

logger = logging.getLogger(__name__)


class FunctionCall:
    replica: FunctionReplica
    request: FunctionRequest
    start: int
    end: Optional[int] = None

    def __init__(self, request, replica, start, end=None):
        self.request = request
        self.replica = replica
        self.start = start
        self.end = end

    @property
    def request_id(self):
        return self.request.request_id


def linear_queue_fet_increase(current_requests: int, max_requests: int) -> float:
    return current_requests / max_requests


class PythonHTTPSimulator(FunctionSimulator):

    def __init__(self, queue: Resource, scale: Callable[[int, int], float], oracle: FittedExecutionTimeOracle):
        self.worker_threads = queue.capacity
        self.queue = queue
        self.scale = scale
        self.oracle = oracle
        self.delay = 0

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        token = self.queue.request()
        yield token  # wait for access

        # because of GIL and Threads, we can easily estimate the additional time caused by concurrent requests to the
        # same Function
        factor = self.scale(self.queue.count, self.queue.capacity)
        _, fet = self.oracle.estimate(env.cluster, replica.pod, SchedulingResult(replica.node.ether_node, 1, []))
        fet = float(fet) * factor
        start = env.now
        replica.node.current_requests.add(request)
        call = FunctionCall(request, replica, start)
        replica.node.all_requests.append(call)
        yield env.timeout(fet)
        end = env.now
        degradation = replica.node.estimate_degradation(start, end)
        delay = max(0, (fet * degradation) - fet)
        yield env.timeout(delay)
        replica.node.set_end(request.request_id, env.now)
        self.queue.release(token)


class PerformanceDegradationSimulator(FunctionSimulator):

    def __init__(self, execution_time_oracle: FittedExecutionTimeOracle):
        self.execution_time_oracle = execution_time_oracle
        self.delay = 0

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        _, fet = self.execution_time_oracle.estimate(env.cluster, replica.pod,
                                                     SchedulingResult(replica.node.ether_node, 1, []))
        fet = float(fet)
        start = env.now
        replica.node.current_requests.add(request)
        call = FunctionCall(request, replica, start)
        replica.node.all_requests.append(call)
        yield env.timeout(fet)
        end = env.now
        degradation = replica.node.estimate_degradation(start, end)
        delay = max(0, (fet * degradation) - fet)
        yield env.timeout(delay)
        replica.node.set_end(request.request_id, env.now)


class PerformanceDegradationFactory(SimulatorFactory):
    def create(self, env: Environment, fn: FunctionDefinition) -> FunctionSimulator:
        mode = fn.labels.get('openfaas', None)
        # return PerformanceDegradationSimulator(FittedExecutionTimeOracle())
        if mode is not None:
            if mode == 'http':
                workers = int(fn.labels['workers'])
                queue = Resource(env=env, capacity=workers)
                return PythonHTTPSimulator(queue, linear_queue_fet_increase, FittedExecutionTimeOracle())
        else:
            return PerformanceDegradationSimulator(FittedExecutionTimeOracle())


def create_fio_function(name='fio'):
    fio_definition = FunctionDefinition(name, 'faas-workloads/fio')
    fio_definition.labels['cpu'] = '0.17294288174214664'
    fio_definition.labels['io'] = '0.8270571182578534'
    fio_definition.labels['gpu'] = '0'
    return fio_definition


def create_resnet50_inference_function(name='resnet50'):
    resnet_definition = FunctionDefinition(name, 'aicg4t1/resnet50-inference')
    resnet_definition.labels['cpu'] = '0.42'
    resnet_definition.labels['io'] = '0.58'
    resnet_definition.labels['gpu'] = '0'
    resnet_definition.labels['openfaas'] = 'http'
    resnet_definition.labels['workers'] = '4'
    return resnet_definition


def create_cpu_load_function(name='cpu_load'):
    cpu_load_definition = FunctionDefinition(name, 'aicg4t1/cpu-load')
    cpu_load_definition.labels['cpu'] = '1'
    cpu_load_definition.labels['io'] = '0'
    cpu_load_definition.labels['gpu'] = '0'
    return cpu_load_definition


class DegradationBenchmark(Benchmark):
    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        self.fio_image = 'faas-workloads/fio'
        self.resnet_image = 'aicg4t1/resnet50-inference'
        self.cpu_load_image = 'aicg4t1/cpu-load'

        containers.put(ImageProperties(self.fio_image, parse_size_string('58M')))
        containers.put(ImageProperties(self.resnet_image, parse_size_string('400M')))
        containers.put(ImageProperties(self.cpu_load_image, parse_size_string('50M')))

        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        resnet_name = 'resnet50'
        fio_function = create_fio_function()
        resnet50_function = create_resnet50_inference_function()
        cpu_load_function = create_cpu_load_function()

        yield from env.faas.deploy(fio_function)
        yield from env.faas.deploy(resnet50_function)
        yield from env.faas.deploy(cpu_load_function)

        logger.info('waiting for replica')
        yield env.process(env.faas.poll_available_replica(resnet_name))
        yield env.process(env.faas.poll_available_replica('fio'))
        yield env.process(env.faas.poll_available_replica('cpu_load'))

        # execute 10 requests in parallel
        logger.info('executing requests')
        # env.process(env.faas.invoke(FunctionRequest('fio')))
        env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        # env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        # env.process(env.faas.invoke(FunctionRequest('cpu_load')))
        clients = 4
        requests_per_client = 50
        for i in range(requests_per_client):
            ps = []
            for j in range(clients):
                ps.append(env.process(env.faas.invoke(FunctionRequest(resnet_name))))

            # wait for invocation processes to finish
            for p in ps:
                yield p

            yield env.timeout(1)


def example_topology() -> Topology:
    t = Topology()
    xeon = create_node(name='eb-xeongpu_1', cpus=4, arch='x86', mem='16Gi', labels={
        'ether.edgerun.io/type': 'vm',
        'ether.edgerun.io/model': 'vm',
        'ether.edgerun.io/capabilities/cuda': '10',
        'ether.edgerun.io/capabilities/gpu': 'turing',
        'ether.edgerun.io/capabilities/vram': '6Gi',
    })
    cell = LANCell(nodes=[xeon], backhaul='internet')
    t.add(cell)
    t.init_docker_registry()

    return t


class FittedSimulationSimulator(FunctionSimulator):

    def __init__(self, oracle: FittedExecutionTimeOracle):
        self.oracle = oracle

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        _, fet = self.oracle.estimate(env.cluster, replica.pod, SchedulingResult(replica.node.ether_node, 1, []))
        yield env.timeout(float(fet))


class FittedEstimationFactory(SimulatorFactory):

    def create(self, env: Environment, fn: FunctionDefinition) -> FunctionSimulator:
        return FittedSimulationSimulator(FittedExecutionTimeOracle())


def main():
    logging.basicConfig(level=logging.DEBUG)

    # TODO: read experiment specification
    topology = example_topology()
    benchmark = DegradationBenchmark()
    env = Environment()
    env.simulator_factory = PerformanceDegradationFactory()
    # env.simulator_factory = FittedEstimationFactory()
    env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
    sim = Simulation(topology, benchmark, env=env)
    sim.run()
    a = sim.env.metrics.extract_dataframe('invocations')
    pass


if __name__ == '__main__':
    main()
