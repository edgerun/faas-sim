from random import random

import numpy as np

from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.main import storage_index
from ext.raith21.topology import HeterogeneousUrbanSensingScenario
from sim.topology import Topology


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
        HeterogeneousUrbanSensingScenario(ether_nodes, storage_index).materialize(topology)
        topology.init_docker_registry()
        return topology


class GlobalIndustrialIoTScenario(TopologyFactory):
    seed: int

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def create(self) -> Topology:
        pass
