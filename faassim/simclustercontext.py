from typing import List, Dict

from core.clustercontext import ClusterContext, BandwidthGraph
from core.model import Node, ImageState


class SimulationClusterContext(ClusterContext):

    def __init__(self, nodes: List[Node], bandwidth_graph: BandwidthGraph):
        self.bandwidth_graph = bandwidth_graph
        self.nodes = nodes
        super().__init__()

        # index all storage nodes
        self.storage_nodes = [node for node in nodes if 'data.skippy.io/storage' in node.labels]

    def list_nodes(self) -> List[Node]:
        return self.nodes

    def get_next_storage_node(self, node: Node) -> str:
        if 'data.skippy.io/storage' in node.labels:
            return node.name
        if not self.storage_nodes:
            return '1_cloud'

        bw = self.get_bandwidth_graph()[node.name]
        storage_node = max(self.storage_nodes, key=lambda n: bw[n.name])

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
