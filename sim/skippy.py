"""
Module that glues simulation concepts to skippy concepts.
"""
import copy
import random
from collections import defaultdict
from typing import List, Dict

from ether.core import Node as EtherNode
from faas.system.core import FunctionContainer
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Node as SkippyNode, Capacity as SkippyCapacity, ImageState, Pod, PodSpec, Container, \
    ResourceRequirements
from skippy.core.storage import StorageIndex
from skippy.core.utils import counter

from sim import docker
from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.topology import LazyBandwidthGraph, DockerRegistry


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
        # FIXME: fix this image state business in skippy
        img_states = {}
        for fn, images in self.container_registry.images.items():
            sizes = {}
            properties = list(images.values())[0]
            for properties in properties:
                arch = properties.arch
                sizes[f'size_{arch}'] = properties.size
            img_states[fn] = ImageState(sizes, 0)
        return img_states

    def retrieve_image_state(self, image_name: str) -> ImageState:
        # FIXME: hacky workaround
        images = self.container_registry.find(image_name)

        if not images:
            raise ValueError('No container image "%s"' % image_name)

        if len(images) == 1 and images[0].arch is None:
            sizes = {
                'x86': images[0].size,
                'arm': images[0].size,
                'arm32v7': images[0].size,
                'aarch64': images[0].size,
                'arm64': images[0].size,
                'amd64': images[0].size
            }
        else:
            sizes = {image.arch: image.size for image in images if image.arch is not None}

        return ImageState(sizes)

    def get_bandwidth_graph(self):
        if self.bw_graph is None:
            self.bw_graph = LazyBandwidthGraph(self.topology)

        return self.bw_graph

    def list_nodes(self) -> List[SkippyNode]:
        if self.nodes is None:
            self.nodes = [to_skippy_node(node) for node in self.topology.get_nodes() if node != DockerRegistry]

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

    labels = dict(node.labels)
    labels['beta.kubernetes.io/arch'] = node.arch

    return SkippyNode(node.name, capacity=capacity, allocatable=allocatable, labels=labels)


pod_counters = defaultdict(counter)


def create_function_pod(fd: SimFunctionDeployment, fn: 'FunctionContainer') -> Pod:
    """
    Creates a new Pod that hosts the given function.
    :param fd: the function deployment to get the deployed function name
    :param fn: the function container to package
    :return: the Pod
    """
    requests = fn.resource_config.get_resource_requirements()
    resource_requirements = ResourceRequirements(requests)

    spec = PodSpec()
    spec.containers = [Container(fn.image, resource_requirements)]
    spec.labels = fn.labels.copy()

    cnt = next(pod_counters[fd.name])
    pod = Pod(f'pod-{fd.name}-{cnt}', 'faas-sim')
    pod.spec = spec

    return pod
