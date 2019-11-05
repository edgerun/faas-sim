import itertools
from typing import Generator, Dict, Callable, List
from core.model import Node, Capacity
from core.utils import parse_size_string

NodeSynthesizer = Generator[Node, None, None]
NodeFactory = List[Callable[[], Node]]


def create_cloud_node(cnt: int) -> Node:
    return create_node(name=f'{cnt}_cloud',
                       cpus=4,
                       mem='8167784Ki',  # kubectl describe node ara-clustercloud1
                       labels={
                           'beta.kubernetes.io/arch': 'amd64',
                           'locality.skippy.io/type': 'cloud'
                       })


def create_tegra_node(cnt: int) -> Node:
    return create_node(name=f'{cnt}_tegra',
                       cpus=4,
                       mem='8047252Ki',
                       labels={
                           'beta.kubernetes.io/arch': 'arm64',
                           'capability.skippy.io/nvidia-cuda': '10',
                           'capability.skippy.io/nvidia-gpu': '',
                           'locality.skippy.io/type': 'edge'
                       })


def create_rpi3_node(cnt: int) -> Node:
    return create_node(name=f'{cnt}_pi',
                       cpus=4,
                       mem='999036Ki',
                       labels={
                           'beta.kubernetes.io/arch': 'arm',
                           'locality.skippy.io/type': 'edge'
                       })


def create_rpi4_node(cnt: int) -> Node:
    return create_node(name=f'{cnt}_rp4',
                       cpus=4,
                       mem='4Gi',
                       labels={
                           'beta.kubernetes.io/arch': 'arm',
                           'locality.skippy.io/type': 'edge'
                       })


def create_node(name: str, cpus: int, mem: str, labels=Dict[str, str]) -> Node:
    capacity = Capacity(cpu_millis=cpus * 1000, memory=parse_size_string(mem))
    allocatable = Capacity(cpu_millis=cpus * 1000, memory=parse_size_string(mem))
    return Node(name, capacity=capacity, allocatable=allocatable, labels=labels)


node_factory_testbed: NodeFactory = [create_cloud_node, create_tegra_node] + [create_rpi3_node] * 4
node_factory_equal: NodeFactory = [create_cloud_node, create_tegra_node, create_rpi3_node]
node_factory_50_percent_cloud: NodeFactory = [create_cloud_node] * 2 + [create_tegra_node, create_rpi3_node]
node_factory_cloud_majority: NodeFactory = [create_cloud_node] * 4 + [create_tegra_node, create_rpi3_node]
node_factory_edge_majority: NodeFactory = [create_cloud_node] + [create_tegra_node] * 2 + [create_rpi3_node] * 4


def node_synthesizer(factory: NodeFactory) -> NodeSynthesizer:
    cnt = 1

    while True:
        # TODO implement different heterogeneity levels
        node = factory[(cnt - 1) % len(factory)](cnt)
        cnt += 1
        yield node


def generate_nodes(node_factory: NodeFactory, node_count: int) -> List[Node]:
    return list(itertools.islice(node_synthesizer(node_factory), node_count))
