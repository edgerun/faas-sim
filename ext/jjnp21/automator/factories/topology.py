import random
import numpy as np

from ext.jjnp21.topology import IndustrialIoTScenario
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.topology import HeterogeneousUrbanSensingScenario
from sim.topology import Topology
from skippy.core.storage import StorageIndex
from ether.core import Connection, Node
from ether.cell import LANCell
from ether.qos import latency


class TopologyFactory:
    def create(self) -> Topology:
        raise Exception('This is just a base class. Please provide an actual implementation instead!')


class RaithHeterogeneousUrbanSensingFactory(TopologyFactory):
    seed: int
    node_count: int

    def __init__(self, node_count: int = 100, seed: int = 42, client_ratio=0) -> None:
        self.seed = seed
        self.node_count = node_count
        self.client_ratio = client_ratio

    def create(self) -> Topology:
        np.random.seed(self.seed)
        random.seed(self.seed)
        devices = generate_devices(self.node_count, cloudcpu_settings)
        ether_nodes = convert_to_ether_nodes(devices)
        topology = Topology()
        storage_index = StorageIndex()
        city = HeterogeneousUrbanSensingScenario(ether_nodes, storage_index, client_ratio=self.client_ratio)
        city.materialize(topology)
        topology.init_docker_registry()
        topology.get_load_balancer_node()
        return topology


class NationDistributedUrbanSensingFactory(TopologyFactory):

    def __init__(self, seed: int = 42, client_ratio=0) -> None:
        self.seed = seed
        self.client_ratio = client_ratio

    def _create_city(self, node_count: int, internet: str, storage_index: StorageIndex, generator_settings,
                     client_ratio: float, city_name: str = 'unknown'):
        devices = generate_devices(node_count, generator_settings)
        ether_nodes = convert_to_ether_nodes(devices)
        for node in ether_nodes:
            node.labels['city'] = city_name
        city = HeterogeneousUrbanSensingScenario(ether_nodes, storage_index, client_ratio=client_ratio,
                                                 internet=internet)
        for client in city.client_nodes:
            client.labels['city'] = city_name
        return city

    def create(self) -> Topology:
        np.random.seed(self.seed)
        random.seed(self.seed)
        topology = Topology()
        storage_index = StorageIndex()
        chicago = self._create_city(100, 'internet_chicago', storage_index, cloudcpu_settings, client_ratio=self.client_ratio, city_name='chicago')
        chicago.materialize(topology)
        new_york = self._create_city(150, 'internet_newyork', storage_index, cloudcpu_settings, client_ratio=self.client_ratio, city_name='newyork')
        new_york.materialize(topology)
        seattle = self._create_city(100, 'internet_seattle', storage_index, cloudcpu_settings, client_ratio=self.client_ratio, city_name='seattle')
        seattle.materialize(topology)

        topology.add_connection(Connection('internet_chicago', 'internet_newyork', latency=31))
        topology.add_connection(Connection('internet_chicago', 'internet_seattle', latency=55))
        topology.add_connection(Connection('internet_seattle', 'internet_newyork', latency=75))
        topology.init_docker_registry()
        topology.get_load_balancer_node()
        return topology


class GlobalDistributedUrbanSensingFactory(TopologyFactory):

    def __init__(self, seed: int = 42, client_ratio=0) -> None:
        self.seed = seed
        self.client_ratio = client_ratio

    def _create_city(self, node_count: int, internet: str, storage_index: StorageIndex, generator_settings,
                     client_ratio: float, city_name: str = 'unknown'):
        devices = generate_devices(node_count, generator_settings)
        ether_nodes = convert_to_ether_nodes(devices)
        for node in ether_nodes:
            node.labels['city'] = city_name
        city = HeterogeneousUrbanSensingScenario(ether_nodes, storage_index, client_ratio=client_ratio,
                                                 internet=internet)
        for client in city.client_nodes:
            client.labels['city'] = city_name
        return city

    def create(self) -> Topology:
        np.random.seed(self.seed)
        random.seed(self.seed)
        topology = Topology()
        storage_index = StorageIndex()
        chicago = self._create_city(100, 'internet_newyork', storage_index, cloudcpu_settings, client_ratio=self.client_ratio, city_name='newyork')
        chicago.materialize(topology)
        new_york = self._create_city(100, 'internet_london', storage_index, cloudcpu_settings, client_ratio=self.client_ratio, city_name='london')
        new_york.materialize(topology)
        seattle = self._create_city(150, 'internet_sydney', storage_index, cloudcpu_settings, client_ratio=self.client_ratio, city_name='sydney')
        seattle.materialize(topology)

        topology.add_connection(Connection('internet_london', 'internet_newyork', latency=86))
        topology.add_connection(Connection('internet_london', 'internet_sydney', latency=253))
        topology.add_connection(Connection('internet_sydney', 'internet_newyork', latency=204))
        topology.init_docker_registry()
        topology.get_load_balancer_node()
        return topology


class GlobalIndustrialIoTScenario(TopologyFactory):
    seed: int

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def create(self) -> Topology:
        topology = Topology()
        scenario1 = IndustrialIoTScenario('iot-1', num_premises=5, clients_per_premise=2, internet='internet_chix')
        scenario1.materialize(topology)
        scenario2 = IndustrialIoTScenario('iot-2', num_premises=2, clients_per_premise=4, internet='internet_nyc')
        scenario2.materialize(topology)
        scenario3 = IndustrialIoTScenario('iot-3', num_premises=3, clients_per_premise=3, internet='internet_sydney')
        scenario3.materialize(topology)
        scenario4 = IndustrialIoTScenario('iot-4', num_premises=6, clients_per_premise=5, internet='internet_stuttgart')
        scenario4.materialize(topology)
        # Todo add proper latencies
        topology.add_connection(Connection('internet_chix', 'internet_nyc', latency_dist=latency.business_isp))
        topology.add_connection(Connection('internet_chix', 'internet_sydney', latency_dist=latency.business_isp))
        topology.add_connection(Connection('internet_chix', 'internet_stuttgart', latency_dist=latency.business_isp))
        topology.add_connection(Connection('internet_nyc', 'internet_sydney', latency_dist=latency.business_isp))
        topology.add_connection(Connection('internet_nyc', 'internet_stuttgart', latency_dist=latency.business_isp))
        topology.add_connection(Connection('internet_sydney', 'internet_stuttgart', latency_dist=latency.business_isp))
        topology.init_docker_registry()
        topology.get_load_balancer_node()

        return topology
