from ether.blocks.nodes import create_nx

from ether.blocks.cells import FiberToExchange
from ether.topology import Topology

from ext.tmueller23.topology.cell import SingleNodeCell


class NxScenario:
    def __init__(self, n=1, internet='internet') -> None:
        super().__init__()
        self.n = n
        self.internet = internet

    def materialize(self, topology: Topology):
        for i in range(self.n):
            topology.add(self.create_singlenode())

    def create_singlenode(self) -> SingleNodeCell:
        return SingleNodeCell(create_nx, backhaul=FiberToExchange(self.internet))
