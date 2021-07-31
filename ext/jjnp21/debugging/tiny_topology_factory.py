import random

import numpy as np
from skippy.core.storage import StorageIndex

from ext.jjnp21.automator.factories.topology import TopologyFactory
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.topology import HeterogeneousUrbanSensingScenario
from sim.topology import Topology


class TinyUrbanSensingTopologyFactory(TopologyFactory):

    def __init__(self, seed: int = 42, client_ratio=0, node_count=20) -> None:
        self.seed = seed
        self.client_ratio = client_ratio
        self.node_count = node_count

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
        chicago = self._create_city(self.node_count, 'internet_chicago', storage_index, cloudcpu_settings,
                                    client_ratio=self.client_ratio, city_name='chicago')
        chicago.materialize(topology)

        topology.init_docker_registry()
        return topology
