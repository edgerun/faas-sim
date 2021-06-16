import logging
import random
from typing import List

from ether.blocks.cells import FiberToExchange, IoTComputeBox
from ether.cell import LANCell, GeoCell, counters, SharedLinkCell, UpDownLink
from ether.core import Node
from ether.scenarios.urbansensing import UrbanSensingScenario, default_cell_density, default_num_cells, \
    default_cloudlet_size
from skippy.core.storage import StorageIndex
from srds import IntegerTruncationSampler

from ext.raith21 import storage
from sim.topology import Topology

logger = logging.getLogger(__name__)


def all_internet_topology(nodes: List[Node]) -> Topology:
    t = Topology()
    for node in nodes:
        cell = LANCell(nodes=[node], backhaul='internet')
        t.add(cell)
    t.init_docker_registry()

    return t


def urban_sensing_topology(nodes: List[Node], storage_index: StorageIndex) -> Topology:
    t = Topology()
    HeterogeneousUrbanSensingScenario(nodes, storage_index).materialize(t)
    t.init_docker_registry()

    return t


class XeonCloudlet(LANCell):

    def __init__(self, xeons: List[Node], xeon_vms_per_rack=5, backhaul=None):
        self.name = None
        self._create_identity()
        self.xeons = xeons
        self.xeon_vms_per_rack = xeon_vms_per_rack
        self.racks = int(len(self.xeons) / self.xeon_vms_per_rack)
        nodes = self.create_nodes()

        super().__init__(nodes, backhaul=backhaul)

    def create_nodes(self) -> List[LANCell]:
        # Explanation for myself:
        # Creates the racks within the cloudlet, where all nodes in a rack are connected by a switch
        # the "nodes" that are returned are the cells/racks. the term cells or rack_cells would be more accurate there
        nodes = []
        rack = []
        for node in self.xeons:
            rack.append(node)
            if len(rack) == self.xeon_vms_per_rack:
                cell = LANCell(rack, backhaul=self.switch)
                nodes.append(cell)
                rack = []
        # this just creates the last smaller rack in case len(node) % rack_count != 0
        if len(rack) > 0:
            cell = LANCell(rack, backhaul=self.switch)
            nodes.append(cell)
        return nodes

    def _create_identity(self):
        if self.name is None:
            self.nr = next(counters['cloudlet'])
            self.name = 'cloudlet_%d' % self.nr
            self.switch = 'switch_%s' % self.name


def parts(a, b):
    """https://stackoverflow.com/a/52698110"""
    q, r = divmod(a, b)
    return [q + 1] * r + [q] * (b - r)


class FasterMobileConnection(UpDownLink):

    def __init__(self, backhaul='internet') -> None:
        super().__init__(250, 250, backhaul)


class HeterogeneousUrbanSensingScenario(UrbanSensingScenario):

    def __init__(self, nodes: List[Node], storage_index: StorageIndex, num_cells=default_num_cells,
                 cell_density=default_cell_density,
                 cloudlet_size=default_cloudlet_size, internet='internet', client_ratio: float = 0) -> None:
        self.nodes = nodes
        self.storage_index = storage_index
        self.xeon_nodes = self._get_xeon_nodes()
        self.rpi3_nodes = self._get_rpi3_nodes()
        self.rpi4_nodes = self._get_rpi4_nodes()
        self.rockpi_nodes = self._get_rockpi_nodes()
        self.tx2_nodes = self._get_tx2_nodes()
        self.nx_nodes = self._get_nx_nodes()
        self.nano_nodes = self._get_nano_nodes()
        self.coral_nodes = self._get_coral_nodes()
        self.nuc_nodes = self._get_nuc_nodes()

        super().__init__(num_cells, cell_density, cloudlet_size, internet)

    def create_city(self) -> GeoCell:
        aot_nodes = []
        aot_nodes.extend(self.create_rpi3_aot_nodes())
        aot_nodes.extend(self.create_rpi4_aot_nodes())
        aot_nodes.extend(self.create_rockpi_aot_nodes())
        nx_nodes = self.nx_nodes
        tx2_nodes = self.tx2_nodes
        nano_nodes = self.nano_nodes
        coral_nodes = self.coral_nodes
        nuc_nodes = self.nuc_nodes
        random.shuffle(aot_nodes)
        neighborhoods = []
        sampler = IntegerTruncationSampler(self.cell_density)
        while len(aot_nodes) > 0:
            size = sampler.sample()

            if size < 4:
                take = random.randint(0, size - 1)
                if take == 0:
                    take = 1
                split = parts(size, take)
            else:
                take = size % 5
                if take == 0:
                    take = 1
                split = parts(size, take)

            while len(split) != 4:
                split.append(0)
            random.shuffle(split)
            choices = random.choices(['tx2', 'nano', 'coral', 'nx'], weights=split, k=take)

            def select_nodes(nodes, n):
                if n > len(nodes):
                    diff = n - len(nodes)
                    n -= diff #so basically if n>len(nodes) then n=len(nodes)
                return nodes[selected_size:], nodes[:selected_size]

            selected_accelerator_nodes = []
            for i in range(take):
                node = choices[i]
                selected_size = split[i]
                if node == 'tx2':
                    tx2_nodes, selected_nodes = select_nodes(tx2_nodes, selected_size)
                    selected_accelerator_nodes.extend(selected_nodes)
                elif node == 'nx':
                    nx_nodes, selected_nodes = select_nodes(nx_nodes, selected_size)
                    selected_accelerator_nodes.extend(selected_nodes)
                elif node == 'nano':
                    nano_nodes, selected_nodes = select_nodes(nano_nodes, selected_size)
                    selected_accelerator_nodes.extend(selected_nodes)
                elif node == 'coral':
                    coral_nodes, selected_nodes = select_nodes(coral_nodes, selected_size)
                    selected_accelerator_nodes.extend(selected_nodes)

            selected_nuc_nodes = []

            if len(nuc_nodes) > 0:
                selected_nuc_nodes.append(nuc_nodes[0])
                nuc_nodes = nuc_nodes[1:]

            selected_aot_nodes = []
            if len(aot_nodes) > 0:
                if size > len(aot_nodes):
                    diff = size - len(aot_nodes)
                    size -= diff

                selected_aot_nodes = aot_nodes[:size]
                aot_nodes = aot_nodes[size:]
            if len(selected_accelerator_nodes) > 0:
                box = IoTComputeBox(selected_accelerator_nodes)
                # This works because the shared link cell materialization recursively resolves this nested list structure
                neighborhood = SharedLinkCell(
                    nodes=[
                        selected_nuc_nodes,
                        selected_aot_nodes,
                        box
                    ],
                    shared_bandwidth=10000,
                    backhaul=FasterMobileConnection(self.internet)
                )
            else:
                neighborhood = SharedLinkCell(
                    nodes=[
                        selected_nuc_nodes,
                        selected_aot_nodes,
                    ],
                    shared_bandwidth=10000,
                    backhaul=FasterMobileConnection(self.internet)
                )

            neighborhoods.append(neighborhood)

        remaining_accelerators = []
        remaining_accelerators.extend(nx_nodes)
        remaining_accelerators.extend(tx2_nodes)
        remaining_accelerators.extend(nano_nodes)
        remaining_accelerators.extend(coral_nodes)

        for accelerator in remaining_accelerators:
            index = random.randint(0, len(neighborhoods) - 1)
            neighborhoods[index].nodes.append([accelerator])

            if len(nuc_nodes) > 0:
                neighborhoods[index].nodes.append([nuc_nodes[0]])
                nuc_nodes = nuc_nodes[1:]

        for nuc_node in nuc_nodes:
            index = random.randint(0, len(neighborhoods) - 1)
            neighborhoods[index].nodes.append([nuc_node])

        def get_first_non_empty_node(neighorbood):
            for n in neighorbood.nodes:
                if type(n) is list and len(n) > 0:
                    if type(n) is IoTComputeBox:
                        return n.nodes[0]
                    return n[0]
            return None

        # init date storages
        for neighborhood in neighborhoods:
            non_empty_node = get_first_non_empty_node(neighborhood)
            if non_empty_node is None:
                continue

            for bucket in storage.bucket_names:
                self.storage_index.mb(bucket, non_empty_node.name)

        for item in storage.data_items:
            self.storage_index.put(item)

        city = GeoCell(self.num_cells, nodes=neighborhoods, density=self.cell_density)

        return city

    def _create_aot_nodes(self, nodes: List[Node], size: int):
        collected = []
        aot_nodes = []
        for node in nodes:
            collected.append(node)
            if len(collected) == size:
                aot_nodes.append(IoTComputeBox(nodes=collected))
                collected = []
        if len(collected) > 0:
            aot_nodes.append(IoTComputeBox(nodes=collected))
        return aot_nodes

    def create_rockpi_aot_nodes(self):
        return self._create_aot_nodes(self.rockpi_nodes, 1)

    def create_rpi4_aot_nodes(self):
        return self._create_aot_nodes(self.rpi4_nodes, 2)

    def create_rpi3_aot_nodes(self):
        return self._create_aot_nodes(self.rpi3_nodes, 3)

    def create_cloudlet(self) -> XeonCloudlet:
        return XeonCloudlet(self.xeon_nodes, self.cloudlet_size[0], backhaul=FiberToExchange(self.internet))

    def _get_xeon_nodes(self) -> List[Node]:
        xeongpus = list(filter(lambda l: 'xeongpu' in l.name, self.nodes))
        xeoncpus = list(filter(lambda l: 'xeoncpu' in l.name, self.nodes))
        xeoncpus.extend(xeongpus)
        return xeoncpus

    def _get_rpi3_nodes(self) -> List[Node]:
        return self._filter_nodes('rpi3')

    def _get_rpi4_nodes(self) -> List[Node]:
        return self._filter_nodes('rpi4')

    def _get_rockpi_nodes(self) -> List[Node]:
        return self._filter_nodes('rockpi')

    def _get_nano_nodes(self) -> List[Node]:
        return self._filter_nodes('nano')

    def _get_tx2_nodes(self) -> List[Node]:
        return self._filter_nodes('tx2')

    def _get_nx_nodes(self) -> List[Node]:
        return self._filter_nodes('nx')

    def _get_nuc_nodes(self) -> List[Node]:
        return self._filter_nodes('nuc')

    def _get_coral_nodes(self) -> List[Node]:
        return self._filter_nodes('coral')

    def _filter_nodes(self, name: str) -> List[Node]:
        return list(filter(lambda n: name in n.name, self.nodes))
