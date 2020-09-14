"""
Module that glues simulation concepts to skippy concepts.
"""
import copy
import random
from typing import List, Dict

from ether.core import Node as EtherNode

from sim.core import Environment
from sim import docker
from sim.topology import LazyBandwidthGraph
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Node as SkippyNode, Capacity as SkippyCapacity, ImageState
from skippy.core.storage import StorageIndex


class SimulationClusterContext(ClusterContext):

    def __init__(self, env: Environment):
        self.env = env

        self.topology = env.topology
        self.container_registry: docker.ContainerRegistry = env.container_registry
        self.bw_graph = None
        self.nodes = None

        super().__init__()

        self.storage_index = env.storage_index or StorageIndex()
        self._storage_nodes = None

    def get_init_image_states(self) -> Dict[str, ImageState]:
        """
        A ImageState dictionary may look like this:
        >>> image_states = {
        >>>     'edgerun/ml-wf-1-pre:0.37': ImageState(size={
        >>>         'arm': 465830200,
        >>>         'arm64': 540391110,
        >>>         'amd64': 533323136
        >>>     }),
        >>>     'edgerun/ml-wf-2-train:0.37': ImageState(size={
        >>>         'arm': 519336111,
        >>>         'arm64': 594174340,
        >>>         'amd64': 550683347
        >>>     }),
        >>> }
        """
        result = dict()

        for image_name, tag_dict in self.container_registry.images.items():
            for tag_name, images in tag_dict.items():
                result[f'{image_name}:{tag_name}'] = ImageState({
                    image.arch: image.size for image in images
                }, 0)

        return result

    def get_bandwidth_graph(self):
        if self.bw_graph is None:
            self.bw_graph = LazyBandwidthGraph(self.topology)

        return self.bw_graph

    def list_nodes(self) -> List[SkippyNode]:
        if self.nodes is None:
            self.nodes = [to_skippy_node(node) for node in self.topology.get_nodes()]

        return self.nodes

    def get_next_storage_node(self, node: SkippyNode) -> str:
        if self.is_storage_node(node):
            return node.name
        if not self.storage_nodes:
            return None

        bw = self.get_bandwidth_graph()[node.name]
        storage_nodes = list(self.storage_nodes.values())
        random.shuffle(storage_nodes)  # make sure you get a random one if bandwidth is the same
        storage_node = max(storage_nodes, key=lambda n: bw[n.name])

        return storage_node.name

    @property
    def storage_nodes(self) -> Dict[str, SkippyNode]:
        if self._storage_nodes is None:
            self._storage_nodes = {node.name: node for node in self.list_nodes() if self.is_storage_node(node)}

        return self._storage_nodes

    def is_storage_node(self, node: SkippyNode):
        return 'data.skippy.io/storage' in node.labels


def to_skippy_node(node: EtherNode) -> SkippyNode:
    """
    Converts an ether Node into a skippy model Node.
    :param node: the node to convert
    :return: the skippy node
    """
    capacity = SkippyCapacity(node.capacity.cpu_millis, node.capacity.memory)
    allocatable = copy.copy(capacity)
    return SkippyNode(node.name, capacity=capacity, allocatable=allocatable, labels=copy.copy(node.labels))

