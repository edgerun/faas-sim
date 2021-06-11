import random
import numpy as np

from ext.jjnp21.topology import IndustrialIoTScenario
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.topology import HeterogeneousUrbanSensingScenario
from sim.topology import Topology
from skippy.core.storage import StorageIndex
from ether.core import Connection
from ether.qos import latency


class TopologyFactory:
    def create(self) -> Topology:
        raise Exception('This is just a base class. Please provide an actual implementation instead!')


class RaithHeterogeneousUrbanSensingFactory(TopologyFactory):
    seed: int
    node_count: int

    def __init__(self, node_count: int = 100, seed: int = 42) -> None:
        self.seed = seed
        self.node_count = node_count

    def create(self) -> Topology:
        np.random.seed(self.seed)
        random.seed(self.seed)
        devices = generate_devices(self.node_count, cloudcpu_settings)
        ether_nodes = convert_to_ether_nodes(devices)
        topology = Topology()
        storage_index = StorageIndex()
        HeterogeneousUrbanSensingScenario(ether_nodes, storage_index).materialize(topology)
        topology.init_docker_registry()
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

        return topology
