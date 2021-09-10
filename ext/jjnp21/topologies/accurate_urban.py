"""
Basic structure:
 function to create parametrized 3G/LTE tower
 function to create datacenter / cloudlet
 function to create parametrized city
 function to
"""
import random
from typing import List

from ether.blocks import nodes as ether_block_nodes
from ether.cell import Cell
from ether.qos import latency
from skippy.core.storage import StorageIndex

from ext.jjnp21.ether_customization.node_generator_settings import *
from ext.jjnp21.ether_customization.towers import RANTower, SmartCityPole
from ext.jjnp21.topology import client_label
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.topology import XeonCloudlet
from sim.core import Node
from sim.topology import Topology, supports_central_load_balancer


class City:
    def __init__(self, cloud_node_count: int, client_count: int, smart_pole_count: int, cell_tower_count: int,
                 five_g_share: float = 0.1, cell_tower_compute_share: float = 0.4, internet='internet', seed: int = 42):
        self.cloud_node_count = cloud_node_count
        self.client_count = client_count
        self.smart_pole_count = smart_pole_count
        self.cell_tower_count = cell_tower_count
        self.cell_tower_compute_share = cell_tower_compute_share
        self.five_g_share = five_g_share
        self.internet = internet
        self.cloud: Cell = None
        self.smart_poles: List[Cell] = []
        self.cell_towers: List[Cell] = []
        self.clients: List[Node] = []
        self._generate()

    def materialize(self, topology: Topology):
        if self.cloud:
            self.cloud.materialize(topology)
        for tower in self.cell_towers:
            tower.materialize(topology)
        for pole in self.smart_poles:
            pole.materialize(topology)

    def _generate(self):
        self._generate_cloud()
        self._generate_cell_towers()
        self._generate_smart_poles()
        self._generate_clients()
        self._attach_clients()

    def _generate_clients(self):
        self.clients = [ether_block_nodes.rpi3() for _ in range(self.client_count)]
        for c in self.clients:
            c.labels[client_label] = 'True'

    def _attach_clients(self):
        # TODO look if a more sophisticated attachment than flat random would make sense
        candidates = self.cell_towers + self.smart_poles
        for client in self.clients:
            choice = random.choice(candidates)
            if isinstance(choice, RANTower):
                choice.radio_nodes.append(client)
            elif isinstance(choice, SmartCityPole):
                choice.nodes.append(client)
            else:
                raise ValueError('Selected choice for client attachment is invalid!')

    def _generate_cloud(self):
        if self.cloud_node_count <= 0:
            return
        nodes = convert_to_ether_nodes(generate_devices(self.cloud_node_count, cloud_settings))
        for node in nodes:
            node.labels[supports_central_load_balancer] = 'True'
        self.cloud = XeonCloudlet(nodes, backhaul=self.internet)

    def _generate_cell_towers(self):
        five_g_count = int(round(self.cell_tower_count * self.five_g_share))
        compute_count = int(round(self.cell_tower_compute_share * self.cell_tower_count))
        is_five_g = []
        has_compute = []
        for i in range(self.cell_tower_count):
            is_five_g.append(i < five_g_count)
            has_compute.append(i < compute_count)
        random.shuffle(is_five_g)
        random.shuffle(has_compute)
        for i in range(self.cell_tower_count):
            tower = None
            compute_nodes = []
            if has_compute[i]:
                compute_nodes = convert_to_ether_nodes(generate_devices(4, edge_intelligence_settings))
            if is_five_g[i]:
                # TODO replace latency with actual 5G latency/bandwidth values
                tower = RANTower([], compute_nodes, self.internet, 1000, latency.wlan)
            else:
                # TODO replace latency with actual LTE latency values
                tower = RANTower([], compute_nodes, self.internet, 100, latency.mobile_isp)
            self.cell_towers.append(tower)

    def _generate_smart_poles(self):
        for _ in range(self.smart_pole_count):
            compute_nodes = convert_to_ether_nodes(generate_devices(2, edge_intelligence_settings))
            pole = SmartCityPole(nodes=compute_nodes, backhaul=self.internet)
            self.smart_poles.append(pole)


def create_city(node_count: int, has_datacenter: bool, client_ratio: float, internet: str,
                 seed: int) -> City:
    # Note that some assumptions about cities are hardcoded here. If you wish to change those you
    # have to change the method signature, or modify them directly
    nc = node_count
    client_node_count = int(round(node_count * client_ratio))
    dc_node_count = 1
    if has_datacenter:
        dc_node_count = 0.3 * node_count
    nc -= dc_node_count
    tower_count = int(round(nc * 0.5))
    pole_count = int(round(nc * 0.25))
    return City(
        cloud_node_count=dc_node_count,
        client_count=client_node_count,
        smart_pole_count=pole_count,
        cell_tower_count=tower_count,
        five_g_share=0.2,
        cell_tower_compute_share=0.25,
        internet=internet,
        seed=seed
    )

def create_smart_pole():
    pass


def create_datacenter():
    pass
