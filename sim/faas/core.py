import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Generator

from ether.core import Node as EtherNode
from faas.context import PlatformContext
from faas.system.core import FunctionRequest, LoadBalancer, FunctionContainer, \
    ResourceConfiguration
from faas.util.constant import function_label, zone_label, api_gateway_type_label
from skippy.core.model import ResourceRequirements

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.context.platform.deployment.service import SimFunctionDeploymentService
from sim.context.platform.replica.model import SimFunctionReplica
from sim.context.platform.replica.service import SimFunctionReplicaService
from sim.core import Environment
from sim.oracle.oracle import FetOracle, ResourceOracle

logger = logging.getLogger(__name__)

Node = EtherNode


class FunctionResourceCharacterization:
    cpu: float
    blkio: float
    gpu: float
    net: float
    ram: float

    def __init__(self, cpu: float, blkio: float, gpu: float, net: float, ram: float):
        self.cpu = cpu
        self.blkio = blkio
        self.gpu = gpu
        self.net = net
        self.ram = ram

    def __len__(self):
        return 5

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


class FunctionCharacterization:

    def __init__(self, image: str, fet_oracle: FetOracle, resource_oracle: ResourceOracle):
        self.image = image
        self.fet_oracle = fet_oracle
        self.resource_oracle = resource_oracle

    def sample_fet(self, host: str) -> Optional[float]:
        return self.fet_oracle.sample(host, self.image)

    def get_resources_for_node(self, host: str) -> FunctionResourceCharacterization:
        return self.resource_oracle.get_resources(host, self.image)


@dataclass
class FunctionSimulatorResponse:
    # response body
    body: str
    # size of the response
    size: int
    # response status code
    code: int
    # timestamp of waiting to be executed
    ts_wait: float
    # timestamp of starting execution
    ts_exec: float
    # raw function execution time, without wait
    fet: float


class SimResourceConfiguration(ResourceConfiguration):
    requests: ResourceRequirements
    limits: Optional[ResourceRequirements]

    def __init__(self, requests: ResourceRequirements = None, limits: ResourceRequirements = None):
        self.requests = requests if requests is not None else ResourceRequirements()
        self.limits = limits

    def get_resource_requirements(self) -> Dict:
        return {
            'cpu': self.requests.requests['cpu'],
            'memory': self.requests.requests['memory']
        }

    def get_resource_limits(self) -> Optional[Dict]:
        if self.limits is not None:
            data = {}
            cpu = self.limits.requests.get('cpu', None)
            memory = self.limits.requests.get('memory', None)
            if cpu is not None:
                data['cpu'] = cpu
            if memory is not None:
                data['memory'] = memory
            return data
        else:
            return None

    @staticmethod
    def create_from_str(cpu: str, memory: str):
        return SimResourceConfiguration(ResourceRequirements.from_str(memory, cpu))


class SimLoadBalancer(LoadBalancer):
    env: Environment

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def get_running_replicas(self, function: str) -> List[SimFunctionReplica]:
        replica_service: SimFunctionReplicaService = self.env.context.replica_service
        return replica_service.get_function_replicas_of_deployment(function, running=True)

    def get_functions(self) -> List[SimFunctionDeployment]:
        deployment_service: SimFunctionDeploymentService = self.env.context.deployment_service
        functions = [d for d in deployment_service.get_deployments() if
                     d.labels.get(function_label) and d.labels.get(function_label) != api_gateway_type_label]
        return functions

    def next_replica(self, request: FunctionRequest) -> SimFunctionReplica:
        raise NotImplementedError


class ContextLoadBalancer(SimLoadBalancer):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env

    def get_running_replicas(self, function: str) -> List[SimFunctionReplica]:
        env: Environment = self.env
        context: PlatformContext = env.context
        replicas = context.replica_service.find_function_replicas_with_labels(labels={
            function_label: function
        }, running=True)
        return replicas

    def get_functions(self) -> List[SimFunctionDeployment]:
        env: Environment = self.env
        context: PlatformContext = env.context
        deployments = context.deployment_service.get_deployments()
        return [deployment for deployment in deployments]

    def next_replica(self, request: FunctionRequest) -> SimFunctionReplica:
        raise NotImplementedError


class LocalizedContextLoadBalancer(SimLoadBalancer):

    def __init__(self, env: Environment, cluster: str):
        super().__init__(env)
        self.cluster = cluster

    def get_running_replicas(self, function: str) -> List[SimFunctionReplica]:
        env: Environment = self.env
        context: PlatformContext = env.context
        replicas = context.replica_service.find_function_replicas_with_labels(labels={
            function_label: function}, node_labels={zone_label: self.cluster}, running=True)
        logger.debug('here')
        all_load_balancers = context.replica_service.find_function_replicas_with_labels(labels={
            function_label: api_gateway_type_label
        })
        other_load_balancers = [l for l in all_load_balancers if l.labels[zone_label] != self.cluster]
        for lb in other_load_balancers:
            other_cluster = lb.labels[zone_label]
            other_replicas = context.replica_service.find_function_replicas_with_labels(labels={
                function_label: function,
            }, node_labels={zone_label: other_cluster}, running=True)
            if len(other_replicas) > 0:
                replicas.append(lb)
        return replicas

    def next_replica(self, request: FunctionRequest) -> SimFunctionReplica:
        raise NotImplementedError()


class LocalizedRoundRobinBalancer(LocalizedContextLoadBalancer):

    def __init__(self, env, cluster):
        super().__init__(env, cluster)
        self.counters = defaultdict(lambda: 0)

    def next_replica(self, request: FunctionRequest) -> SimFunctionReplica:
        replicas = self.get_running_replicas(request.name)
        if request.headers is not None:
            replicas = [r for r in replicas if r.labels[function_label] != api_gateway_type_label]
        i = self.counters[request.name] % len(replicas)
        self.counters[request.name] = (i + 1) % len(replicas)

        replica = replicas[i]

        return replica


class RoundRobinLoadBalancer(SimLoadBalancer):

    def __init__(self, env) -> None:
        super().__init__(env)
        self.counters = defaultdict(lambda: 0)

    def next_replica(self, request: FunctionRequest) -> SimFunctionReplica:
        replicas = self.get_running_replicas(request.name)
        i = self.counters[request.name] % len(replicas)
        self.counters[request.name] = (i + 1) % len(replicas)

        replica = replicas[i]

        return replica


class FunctionSimulator(abc.ABC):

    def deploy(self, env: Environment, replica: SimFunctionReplica):
        yield env.timeout(0)

    def startup(self, env: Environment, replica: SimFunctionReplica):
        yield env.timeout(0)

    def setup(self, env: Environment, replica: SimFunctionReplica):
        yield env.timeout(0)

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, Optional[FunctionSimulatorResponse]]:
        yield env.timeout(0)
        return FunctionSimulatorResponse(request.body, 150, 200, 0, 0, 0)

    def teardown(self, env: Environment, replica: SimFunctionReplica):
        yield env.timeout(0)


class SimulatorFactory:

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        raise NotImplementedError
