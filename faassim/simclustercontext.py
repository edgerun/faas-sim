from typing import List, Dict

from core.clustercontext import ClusterContext, BandwidthGraph
from core.model import Node, ImageState


class SimulationClusterContext(ClusterContext):

    def __init__(self, nodes: List[Node], bandwidth_graph: BandwidthGraph):
        self.bandwidth_graph = bandwidth_graph
        self.nodes = nodes
        super().__init__()

    def list_nodes(self) -> List[Node]:
        return self.nodes

    def get_next_storage_node(self, node: Node) -> str:
        # TODO maybe extend to simulate multiple data nodes and find the one next to the given node
        return '1_cloud'

    def get_init_image_states(self) -> Dict[str, ImageState]:
        # TODO maybe synth other images?
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-1-pre/tags/0.33/
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-2-train/tags/0.33/
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-3-serve/tags/0.33/
        return {
            'alexrashed/ml-wf-1-pre:0.33': ImageState(size={
                'arm': 461473086,
                'arm64': 538015840,
                'amd64': 530300745
            }),
            'alexrashed/ml-wf-2-train:0.33': ImageState(size={
                'arm': 506029298,
                'arm64': 582828211,
                'amd64': 547365470
            }),
            'alexrashed/ml-wf-3-serve:0.33': ImageState(size={
                'arm': 506769993,
                'arm64': 585625232,
                'amd64': 585928717
            }),
            'alexrashed/ml-wf-1-pre:0.34': ImageState(size={
                'arm': 465830200,
                'arm64': 540391110,
                'amd64': 533323136
            }),
            'alexrashed/ml-wf-2-train:0.34': ImageState(size={
                'arm': 519336111,
                'arm64': 594174340,
                'amd64': 550683347
            }),
            'alexrashed/ml-wf-3-serve:0.34': ImageState(size={
                'arm': 511888808,
                'arm64': 590989596,
                'amd64': 589680790
            }),
            'alexrashed/ml-wf-1-pre:0.35': ImageState(size={
                'arm': 465830200,
                'arm64': 540391110,
                'amd64': 533323136
            }),
            'alexrashed/ml-wf-2-train:0.35': ImageState(size={
                'arm': 519336111,
                'arm64': 594174340,
                'amd64': 550683347
            }),
            'alexrashed/ml-wf-3-serve:0.35': ImageState(size={
                'arm': 511888808,
                'arm64': 590989596,
                'amd64': 589680790
            })
        }

    def get_bandwidth_graph(self) -> BandwidthGraph:
        return self.bandwidth_graph
