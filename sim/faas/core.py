import abc
import enum
import logging
from collections import defaultdict
from typing import List, Dict, NamedTuple, Optional

from ether.core import Node as EtherNode
from ether.util import parse_size_string
from skippy.core.model import Pod

from sim.core import Environment, NodeState
from sim.oracle.oracle import FetOracle, ResourceOracle

logger = logging.getLogger(__name__)

Node = EtherNode


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

    def __init__(self, cpu_millis: int = 1 * 1000, memory: int = 1 * 1024 * 1024):
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
        return Resources(int(cpu.rstrip('m')), parse_size_string(memory))


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


class FunctionImage:
    # the manifest list (docker image) name
    image: str

    def __init__(self, image: str):
        self.image = image


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


class ResourceConfiguration(abc.ABC):

    def get_resource_requirements(self) -> Dict: ...


class KubernetesResourceConfiguration(ResourceConfiguration):
    requests: Resources

    def __init__(self, requests: Resources = None):
        self.requests = requests if requests is not None else Resources()

    def get_resource_requirements(self) -> Dict:
        return {
            'cpu': self.requests.cpu,
            'memory': self.requests.memory
        }

    @staticmethod
    def create_from_str(cpu: str, memory: str):
        return KubernetesResourceConfiguration(Resources.from_str(memory, cpu))


class Function:
    name: str
    fn_images: List[FunctionImage]
    # TODO cascading labeling
    labels: Dict[str, str]

    def __init__(self, name: str, fn_images: List[FunctionImage], labels: Dict[str, str] = None):
        self.fn_images = fn_images
        self.name = name
        self.labels = labels if labels is not None else {}

    def get_image(self, image: str) -> Optional[FunctionImage]:
        for fn_image in self.fn_images:
            if fn_image.image == image:
                return fn_image
        return None


class FunctionContainer:
    fn_image: FunctionImage
    resource_config: ResourceConfiguration
    labels: Dict[str, str]

    def __init__(self, fn_image: FunctionImage, resource_config: ResourceConfiguration = None,
                 labels: Dict[str, str] = None):
        self.fn_image = fn_image
        self.resource_config = resource_config if resource_config is not None else KubernetesResourceConfiguration()
        self.labels = labels if labels is not None else {}

    @property
    def image(self):
        return self.fn_image.image

    def get_resource_requirements(self):
        return self.resource_config.get_resource_requirements()


class ScalingConfiguration:
    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 1
    scale_zero: bool = False

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


class FunctionDeployment:
    fn: Function
    fn_containers: List[FunctionContainer]
    scaling_config: ScalingConfiguration
    # used to determine which function to take when scaling
    ranking: DeploymentRanking

    def __init__(self, fn: Function, fn_containers: List[FunctionContainer], scaling_config: ScalingConfiguration,
                 deployment_ranking: DeploymentRanking = None):
        self.fn = fn
        self.fn_containers = fn_containers
        self.scaling_config = scaling_config
        if deployment_ranking is None:
            self.ranking = DeploymentRanking([x.image for x in self.fn.fn_images])
        else:
            self.ranking = deployment_ranking

    def get_selected_service(self):
        return self.fn.get_image(self.ranking.get_first())

    def get_services(self):
        return list(map(lambda image: self.fn.get_image(image), self.ranking.images))

    def get_containers(self):
        return [self.get_container(image) for image in self.ranking.images]

    def get_container(self, image: str) -> Optional[FunctionContainer]:
        for fn_image in self.fn_containers:
            if fn_image.image == image:
                return fn_image
        return None

    @property
    def name(self):
        return self.fn.name


class FunctionReplica:
    """
    A function replica is an instance of a function running on a specific node.
    """
    function: FunctionDeployment
    container: FunctionContainer
    node: NodeState
    pod: Pod
    state: FunctionState = FunctionState.CONCEIVED

    simulator: 'FunctionSimulator' = None

    @property
    def fn_name(self):
        return self.function.name

    @property
    def image(self):
        return self.container.image


class FunctionRequest:
    request_id: int
    name: str
    size: float = None
    load_balancer = None
    client_node: EtherNode = None

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


class FaasSystem(abc.ABC):

    @abc.abstractmethod
    def deploy(self, fn: FunctionDeployment): ...

    @abc.abstractmethod
    def invoke(self, request: FunctionRequest): ...

    @abc.abstractmethod
    def remove(self, fn: FunctionDeployment): ...

    @abc.abstractmethod
    def get_deployments(self) -> List[FunctionDeployment]: ...

    @abc.abstractmethod
    def get_function_index(self) -> Dict[str, FunctionContainer]: ...

    @abc.abstractmethod
    def get_replicas(self, fn_name: str, state=None) -> List[FunctionReplica]: ...

    @abc.abstractmethod
    def scale_down(self, function_name: str, remove: int): ...

    @abc.abstractmethod
    def scale_up(self, function_name: str, replicas: int): ...

    @abc.abstractmethod
    def discover(self, function: FunctionContainer) -> List[FunctionReplica]: ...

    @abc.abstractmethod
    def suspend(self, function_name: str): ...

    @abc.abstractmethod
    def poll_available_replica(self, fn: str, interval=0.5): ...


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

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        raise NotImplementedError
