import random
from typing import List, Dict

from core.clustercontext import ClusterContext, BandwidthGraph
from core.model import Node, ImageState


class SimulationClusterContext(ClusterContext):

    def __init__(self, nodes: List[Node], bandwidth_graph: BandwidthGraph):
        self.bandwidth_graph = bandwidth_graph
        random.shuffle(list(nodes))
        self.nodes = nodes
        self.node_index = {node.name: node for node in nodes}
        super().__init__()

        # index all storage nodes
        self.storage_nodes = {node.name: node for node in nodes if 'data.skippy.io/storage' in node.labels}
        self.storage_node_index: Dict[Node, Node] = dict()

    def list_nodes(self) -> List[Node]:
        return self.nodes

    def get_node(self, name: str) -> Node:
        return self.node_index.get(name)

    def get_next_storage_node(self, node: Node) -> str:
        if 'data.skippy.io/storage' in node.labels:
            return node.name
        if not self.storage_nodes:
            return '1_cloud'

        if node in self.storage_node_index:
            return self.storage_node_index[node].name

        bw = self.get_bandwidth_graph()[node.name]
        storage_nodes = list(self.storage_nodes.values())
        random.shuffle(storage_nodes)  # make sure you get a random one if bandwidth is the same
        storage_node = max(storage_nodes, key=lambda n: bw[n.name])

        self.storage_node_index[node] = storage_node
        return storage_node.name

    def get_init_image_states(self) -> Dict[str, ImageState]:
        # TODO maybe synth other images?
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-1-pre/tags/0.36/
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-2-train/tags/0.36/
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-3-serve/tags/0.36/
        return {
            'alexrashed/ml-wf-1-pre:0.37': ImageState(size={
                'arm': 465830200,
                'arm64': 540391110,
                'amd64': 533323136
            }),
            'alexrashed/ml-wf-2-train:0.37': ImageState(size={
                'arm': 519336111,
                'arm64': 594174340,
                'amd64': 550683347
            }),
            'alexrashed/ml-wf-3-serve:0.37': ImageState(size={
                'arm': 511888808,
                'arm64': 590989596,
                'amd64': 589680790
            })
        }

    def get_bandwidth_graph(self) -> BandwidthGraph:
        return self.bandwidth_graph
