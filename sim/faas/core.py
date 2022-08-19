import abc
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Generator

from ether.core import Node as EtherNode
from faas.system.core import FunctionRequest, LoadBalancer, FunctionReplica, FunctionDeployment, FunctionContainer, \
    Function, FunctionReplicaState, ScalingConfiguration, ResourceConfiguration, FunctionResponse
from skippy.core.model import Pod, ResourceRequirements

from sim.core import Environment, NodeState
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


class DeploymentRanking:
    # TODO probably better to remove default/enable default for one image
    images: List[str]

    # TODO probably removable after moving decision on which node to deploy pod to user
    # percentages of scaling per image, can be used to hinder scheduler to overuse expensive resources (i.e. tpu)
    function_factor: Dict[str, float]

    def __init__(self, images: List[str], function_factor: Dict[str, float] = None):
        self.images = images
        self.function_factor = function_factor if function_factor is not None else {image: 1 for image in images}

    def set_first(self, image: str):
        index = self.images.index(image)
        updated = self.images[:index] + self.images[index + 1:]
        self.images = [image] + updated

    def get_first(self):
        return self.images[0]


class SimScalingConfiguration:
    scaling_config: ScalingConfiguration

    def __init__(self, scaling_config: ScalingConfiguration = None):
        self.scaling_config = ScalingConfiguration()
        if scaling_config is None:
            self.scaling_config = ScalingConfiguration()

    @property
    def scale_min(self):
        return self.scaling_config.scale_min

    @property
    def scale_max(self):
        return self.scaling_config.scale_max

    @property
    def scale_zero(self):
        return self.scaling_config.scale_zero

    @property
    def scale_factor(self):
        return self.scale_factor

    # average requests per second threshold for scaling
    rps_threshold: int = 20

    # window over which to track the average rps
    alert_window: int = 50  # TODO currently not supported by FaasRequestScaler

    # seconds the rps threshold must be violated to trigger scale up
    rps_threshold_duration: int = 10

    # target average cpu utilization of all replicas, used by HPA
    target_average_utilization: float = 0.5

    # target average rps over all replicas, used by AverageFaasRequestScaler
    target_average_rps: int = 200

    # target of maximum requests in queue
    target_queue_length: int = 75

    target_average_rps_threshold = 0.1


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


class SimFunctionDeployment(FunctionDeployment):
    scaling_config: SimScalingConfiguration
    # used to determine which function to take when scaling
    ranking: DeploymentRanking

    def __init__(self, fn: Function, fn_containers: List[FunctionContainer], scaling_config: SimScalingConfiguration,
                 deployment_ranking: DeploymentRanking = None):
        super().__init__(fn, fn_containers, scaling_config, deployment_ranking)
        self.scaling_config = scaling_config
        if deployment_ranking is None:
            self.ranking = DeploymentRanking([x.image for x in self.fn.fn_images])
        else:
            self.ranking = deployment_ranking

    def get_selected_service(self):
        return self.fn.get_image(self.ranking.get_first())


class SimFunctionReplica(FunctionReplica):
    """
    A function replica is an instance of a function running on a specific node.
    """
    function: SimFunctionDeployment
    container: FunctionContainer
    node: NodeState
    pod: Pod
    state: FunctionReplicaState = FunctionReplicaState.CONCEIVED

    simulator: 'FunctionSimulator' = None

    @property
    def fn_name(self):
        return self.function.name

    @property
    def image(self):
        return self.container.image


class SimLoadBalancer(LoadBalancer):
    env: Environment
    replicas: Dict[str, List[SimFunctionReplica]]

    def __init__(self, env, replicas) -> None:
        super().__init__()
        self.env = env
        self.replicas = replicas

    def get_running_replicas(self, function: str):
        return [replica for replica in self.replicas[function] if replica.state == FunctionReplicaState.RUNNING]

    def next_replica(self, request: FunctionRequest) -> SimFunctionReplica:
        raise NotImplementedError


class RoundRobinLoadBalancer(SimLoadBalancer):

    def __init__(self, env, replicas) -> None:
        super().__init__(env, replicas)
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
        None, None, Optional[FunctionResponse]]:
        yield env.timeout(0)
        return None

    def teardown(self, env: Environment, replica: SimFunctionReplica):
        yield env.timeout(0)


class SimulatorFactory:

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        raise NotImplementedError
