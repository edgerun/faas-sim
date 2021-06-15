from typing import List

from ether.blocks import nodes
from ether.blocks.cells import IoTComputeBox, MobileConnection, BusinessIsp
from ether.blocks.nodes import create_rpi3_node, create_vm_node
from ether.cell import LANCell, SharedLinkCell, UpDownLink, counters, GeoCell
from ether.core import Connection, Node
from sim.topology import Topology
from srds import ConstantSampler
from ether.qos import latency

client_label = 'beta.osmotic.io/client'

default_num_cells = 1
default_cell_density = ConstantSampler(10)
default_client_per_premise = 1


class SmallCloudlet(LANCell):
    def __init__(self, server_per_rack=5, racks=1, backhaul=None) -> None:
        self.racks = racks
        self.server_per_rack = server_per_rack

        nodes = [self._create_rack] * racks

        super().__init__(nodes, backhaul=backhaul)

    def _create_identity(self):
        self.nr = next(counters['cloudlet'])
        self.name = 'cloudlet_%d' % self.nr
        self.switch = 'switch_%s' % self.name

    def _create_rack(self):
        return LANCell([create_vm_node] * self.server_per_rack, backhaul=self.switch)
        # Todo: Replace this (create_vm_node) with a xeon node. right now it is a generic 'cloud_vm', which the
        # FET oracle does not recognize. honestly no idea where it pulls values for those from


class ClientGroup(LANCell):
    def __init__(self, clients=1, backhaul=None) -> None:
        self.clients = clients
        self.rpis = [create_rpi3_node() for _ in range(clients)]
        for rpi in self.rpis:
            rpi.labels[client_label] = 'True'
        super().__init__(self.rpis, backhaul=backhaul)


class WeakBusinessIsp(UpDownLink):
    def __init__(self, backhaul='internet') -> None:
        super().__init__(100, 20, backhaul, latency.business_isp)


class IndustrialIoTScenario:
    def __init__(self, name: str, clients_per_premise=default_client_per_premise, num_premises=default_num_cells,
                 premises_density=default_cell_density,
                 internet='internet') -> None:
        """
        The IIoT scenarios with several factories, that have a factory floor with IoT devices and a on-premises managed
        cloudlet.

        :param num_premises: the number of premises, each premises is a factory with a floor and a cloudlet
        :param premises_density: currently not used, but the idea is that the total number of devices on a premises vary
        according to the parameter. but it's unclear how the total number of devices should be split among the floor and
        the cloudlet.
        :param internet:
        """
        super().__init__()
        self.name = name
        self.num_premises = num_premises
        self.premises_density = premises_density
        self.clients_per_premise = clients_per_premise
        self.internet = internet
        self.clients_gateway_mapping = {}
        self.api_gateways = []

    def materialize(self, topology: Topology):
        for i in range(self.num_premises):
            floor_compute = IoTComputeBox(nodes=[nodes.nuc, nodes.tx2])
            floor_iot = SharedLinkCell(nodes=[nodes.rpi3] * 3)

            factory = LANCell([floor_compute, floor_iot], backhaul=WeakBusinessIsp(self.internet))
            factory.materialize(topology)

            cloudlet = SmallCloudlet(1, 1, backhaul=UpDownLink(10000, 10000, backhaul=factory.switch))
            cloudlet.materialize(topology)
            clients = ClientGroup(self.clients_per_premise, backhaul=MobileConnection(backhaul=factory.switch))
            clients.materialize(topology)


def get_client_nodes(topology: Topology) -> List[Node]:
    clients = []
    for node in topology.get_nodes():
        if node.labels.get(client_label, None) is not None:
            clients.append(node)
    return clients


def get_non_client_nodes(topology: Topology) -> List[Node]:
    return [node for node in topology.get_nodes() if node.labels.get(client_label, None) is None]

