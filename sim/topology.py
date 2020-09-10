from typing import Optional

import ether.topology
from ether.core import Node

DockerRegistry = Node('dockerhub.com')


class Topology(ether.topology.Topology):

    def find_node(self, node_name: str) -> Optional[Node]:
        for node in self.get_nodes():
            if node.name == node_name:
                return node

        return None
