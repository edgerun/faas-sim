from typing import List

from ether.cell import Cell, UpDownLink, SharedLinkCell, LANCell, Host
from ether.cell import counters
from ether.core import Node, Link, Connection
from ether.qos import latency
from ether.topology import Topology


class WLANCell(Cell):
    def __init__(self, nodes, backhaul=None):
        super().__init__(nodes=nodes, backhaul=backhaul)

    def _create_identity(self):
        self.nr = next(counters['wlan'])
        self.name = 'wlan_%d' % self.nr
        self.switch = 'ap_%s' % self.name

    def materialize(self, topology: Topology, parent=None):
        self._create_identity()

        for node in self.nodes:
            # We manually create a host here, since this allows us to set the latency distribution.
            # Also, the Host uses a default bandwidth of 1000MBps which is a reasonable estimate for 802.11ac
            h = Host(node, backhaul=self.switch)
            # h.materialize(topology, self, latency.lan)
            h.materialize(topology, self, latency.wlan)

        if self.backhaul:
            if isinstance(self.backhaul, UpDownLink):
                uplink = Link(self.backhaul.bw_up, tags={'type': 'uplink', 'name': 'up_%s' % self.name})
                downlink = Link(self.backhaul.bw_down, tags={'type': 'downlink', 'name': 'down_%s' % self.name})

                topology.add_connection(Connection(self.switch, uplink, latency_dist=self.backhaul.latency_dist),
                                        directed=True)
                topology.add_connection(Connection(downlink, self.switch), directed=True)

                topology.add_connection(Connection(self.backhaul.backhaul, downlink,
                                                   latency_dist=self.backhaul.latency_dist), directed=True)
                topology.add_connection(Connection(uplink, self.backhaul.backhaul), directed=True)

            else:
                topology.add_connection(Connection(self.switch, self.backhaul, latency_dist=latency.mobile_isp))


"""
General Idea here:
Client nodes can have different kinds of connection:
 - Via a WLAN cell with a direct uplink: This represents a typical "office" or home wifi situation
 - Via RANTower: This represents a typical mobile connection. Tower can be 5G or LTE
 - Via SmartCityPole: This is a wifi connection simulating something like a huawei polestar, where there's compute on the pole
    and the pole itself has a mobile internet connection
"""


class SmartCityPole(WLANCell):
    pass


class RANTower(Cell):

    def __init__(self, radio_nodes: List[Node], local_nodes: List[Node], backhaul=None, shared_radio_bandwidth=100,
                 radio_latency_dist=latency.mobile_isp):
        super().__init__(radio_nodes.copy().extend(local_nodes), backhaul=backhaul)
        self.radio_nodes = radio_nodes
        self.local_nodes = local_nodes
        self.shared_radio_bandwidth = shared_radio_bandwidth
        self.radio_latency_dist = radio_latency_dist

    def _create_identity(self):
        self.nr = next(counters['ran'])
        self.name = 'ran_%d' % self.nr
        self.switch = 'switch_%s' % self.name
        self.local_compute = LANCell(self.local_nodes, self.switch)
        self.radio = SharedLinkCell(self.radio_nodes,
                                    backhaul=UpDownLink(self.shared_radio_bandwidth, self.shared_radio_bandwidth,
                                                        backhaul=self.switch,
                                                        latency_dist=self.radio_latency_dist))
        # todo don't use a shared link cell here
        # we do not want a shared up-down link. First of all that kind of "shared bandwidth" is only semi-true
        # second of all this will once again create a large number of shared flows leading to long calculation times

    def materialize(self, topology: Topology, parent=None):
        self._create_identity()
        self.local_compute.materialize(topology)
        self.radio.materialize(topology)

        if self.backhaul:
            if isinstance(self.backhaul, UpDownLink):
                uplink = Link(self.backhaul.bw_up, tags={'type': 'uplink', 'name': 'up_%s' % self.name})
                downlink = Link(self.backhaul.bw_down, tags={'type': 'downlink', 'name': 'down_%s' % self.name})

                topology.add_connection(Connection(self.switch, uplink, latency_dist=self.backhaul.latency_dist), True)
                topology.add_connection(Connection(downlink, self.switch), True)

                topology.add_connection(Connection(self.backhaul.backhaul, downlink,
                                                   latency_dist=self.backhaul.latency_dist), directed=True)
                topology.add_connection(Connection(uplink, self.backhaul.backhaul), directed=True)

            else:
                topology.add_connection(Connection(self.switch, self.backhaul))
