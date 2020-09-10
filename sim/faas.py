import enum
from collections import defaultdict
from typing import List, Dict, NamedTuple

from ether.util import parse_size_string

from sim.core import Environment
from skippy.core.utils import counter


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
        self.cpu_millis = cpu_millis

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
    labels: Dict[str, str] = {}

    requests: Resources = Resources()

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 20
    scale_zero: bool = False

    def __init__(self, name, image) -> None:
        super().__init__()
        self.name = name
        self.image = image


class FunctionReplica:
    """
    A function replica is an instance of a function running on a specific node.
    """
    function: FunctionDefinition
    node: str
    state: FunctionState


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



class FaasSystem:
    """

    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.functions = dict()
        self.replicas = defaultdict(list)

    def get_replicas(self, function_name: str, state=None) -> List[FunctionReplica]:
        if state is None:
            return self.replicas[function_name]

        return [replica for replica in self.replicas[function_name] if replica.state == state]

    def deploy(self, fn: FunctionDefinition):
        if self.functions[fn.name]:
            raise ValueError('function already deployed')

        # TODO: create function instance
        # TODO: create replica

        pass
