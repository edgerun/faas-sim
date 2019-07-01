from typing import List, Dict

from core.clustercontext import ClusterContext
from core.model import Node, ImageState


class SimulationClusterContext(ClusterContext):

    def retrieve_image_state(self, image_name: str) -> ImageState:
        # TODO return the image states for the image names used by the pod synthesizer
        pass

    def list_nodes(self) -> List[Node]:
        # TODO
        return [Node('test')]