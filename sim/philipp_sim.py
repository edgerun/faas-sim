import logging

from skippy.core.model import SchedulingResult

from faassim.oracle.oracle import FittedExecutionTimeOracle
from sim.cli.degradation import FunctionCall
from sim.core import Environment
from sim.docker import pull as docker_pull
from sim.faas import FunctionReplica, FunctionRequest, FunctionSimulator, SimulatorFactory, \
    FunctionDefinition

logger = logging.getLogger(__name__)


def gpu_multiplicator_cpu(need_cpu: float, cpu_usage: float, cores: int):
    used_cores = cpu_usage * cores
    need_cores = need_cpu * cores
    if cores - used_cores >= need_cores:
        return 0
    else:
        return 0.25


def gpu_multiplicator_gpu(need_gpu: float, gpu_usage: float):
    if need_gpu + gpu_usage < 1:
        return 0.10
    else:
        return 0.5


def cpu_multiplicator_cpu(need_cpu: float, cpu_usage: float, cores: int):
    used_cores = cpu_usage * cores
    need_cores = need_cpu * cores
    if cores - used_cores >= need_cores:
        return 0
    else:
        return 0.25


def gpu_multi_same_pod_cpu(need_cpu: float):
    return need_cpu


class GPUFunctionSimulator(FunctionSimulator):

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield from docker_pull(env, replica.function.image, replica.node.ether_node)

    def startup(self, env: Environment, replica: FunctionReplica):
        return super().startup(env, replica)

    def setup(self, env: Environment, replica: FunctionReplica):
        return super().setup(env, replica)

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        # which concurrent requests are running
        # are they running on same replica
        # if yes, then add CPU overhead caused by single thread contention

        # are they running on same node but a different pod
        # if yes then add overhead in case CPU contention occurs

        # if yes and GPU Bound, add small increase due to GPU contention
        return super().invoke(env, replica, request)

    def teardown(self, env: Environment, replica: FunctionReplica):
        return super().teardown(env, replica)




class GPUSimulatorFactory(SimulatorFactory):
    def create(self, env: Environment, fn: FunctionDefinition) -> FunctionSimulator:
        return GPUFunctionSimulator()


class CPUFunctionSimulator(FunctionSimulator):

    def deploy(self, env: Environment, replica: FunctionReplica):
        yield from docker_pull(env, replica.function.image, replica.node.ether_node)

    def startup(self, env: Environment, replica: FunctionReplica):
        return super().startup(env, replica)

    def setup(self, env: Environment, replica: FunctionReplica):
        return super().setup(env, replica)

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        return super().invoke(env, replica, request)

    def teardown(self, env: Environment, replica: FunctionReplica):
        return super().teardown(env, replica)


class PhilippsSimulatorFactory(SimulatorFactory):
    def create(self, env: Environment, fn: FunctionDefinition) -> FunctionSimulator:
        process_type = 'process_type'
        if fn.labels[process_type] == 'cpu':
            return CPUFunctionSimulator()
        elif fn.labels[process_type] == 'gpu':
            return GPUFunctionSimulator()
        else:
            raise AttributeError(f'Unknown process type for fn: {fn.name}, {fn.labels[process_type]}')
