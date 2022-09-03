from typing import List

from faas.util.constant import client_role_label

from sim.faas.core import Node
from sim.topology import Topology


def find_clients(topology: Topology) -> List[Node]:
    return [x for x in topology.get_nodes() if x.labels.get(client_role_label) is not None]
