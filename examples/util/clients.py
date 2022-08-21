from typing import List

from sim.core import Node
from sim.topology import Topology


def find_clients(topology: Topology, client='rpi3') -> List[Node]:
    return [x for x in topology.get_nodes() if client in x.name]

