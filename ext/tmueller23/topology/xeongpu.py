from ether.blocks.cells import FiberToExchange
from ether.topology import Topology
from ether.vis import draw_basic
import matplotlib.pyplot as plt

from ext.raith21.etherdevices import create_xeongpu
from ext.tmueller23.topology.cell import SingleNodeCell


class XeonScenario:
    def __init__(self, n=1, internet='internet') -> None:
        super().__init__()
        self.n = n
        self.internet = internet

    def materialize(self, topology: Topology):
        for i in range(self.n):
            topology.add(self.create_singlenode())

    def create_singlenode(self) -> SingleNodeCell:
        return SingleNodeCell(create_xeongpu, backhaul=FiberToExchange(self.internet))


if __name__ == '__main__':
    t = Topology()
    XeonScenario(10).materialize(t)
    draw_basic(t)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()  # display
