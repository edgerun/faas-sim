import time
from typing import Set, Optional

import simpy
from ether.core import Node as EtherNode

Node = EtherNode


class NodeState:
    """
    Holds simulation specific runtime knowledge about a node. For example, what docker images it has already pulled.
    """
    docker_images: Set

    def __init__(self) -> None:
        super().__init__()
        self.ether_node = None
        self.skippy_node = None
        self.docker_images = set()

    @property
    def name(self):
        return self.ether_node.name

    @property
    def arch(self):
        return self.ether_node.arch


class SimulationTimeoutError(BaseException):
    pass


class Environment(simpy.Environment):
    cluster: 'SimulationClusterContext'

    def __init__(self, initial_time=0):
        super().__init__(initial_time)
        self.faas = None
        self.topology = None
        self.storage_index = None
        self.benchmark = None
        self.cluster = None
        self.registry = None
        self.metrics = None
        self.node_states = dict()

    def get_node_state(self, name: str) -> Optional[NodeState]:
        """
        Lazy-loads a NodeState for the given node name by looking for it in the topology.

        :param name: the node name
        :return: a new or chached NodeState
        """
        if name in self.node_states:
            return self.node_states[name]

        ether_node = self.topology.find_node(name) if self.topology else None
        skippy_node = self.cluster.get_node(name) if self.cluster else None

        node_state = NodeState()
        node_state.env = self
        node_state.ether_node = ether_node
        node_state.skippy_node = skippy_node

        self.node_states[name] = node_state
        return node_state


def timeout_listener(env, started, max_time, interval=1):
    while True:
        yield env.timeout(interval)

        if (time.time() - started) > max_time:
            raise SimulationTimeoutError()
