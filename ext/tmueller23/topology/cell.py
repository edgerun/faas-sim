from ether.blocks.nodes import create_nuc_node
from ether.cell import LANCell, counters


class SingleNodeCell(LANCell):
    def __init__(self, node, backhaul=None) -> None:
        self.node = node
        nodes = [node, create_nuc_node]

        super().__init__(nodes, backhaul=backhaul)

    def _create_identity(self):
        self.nr = next(counters['cloudlet'])
        self.name = 'singlenode_%d' % self.nr
        self.switch = 'switch_%s' % self.name
