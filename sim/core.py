import time
from typing import Set, Optional

import simpy
from ether.core import Node as EtherNode

from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Node as SkippyNode

Node = EtherNode


class NodeState:
    """
    Holds simulation specific runtime knowledge about a node. For example, what docker images it has already pulled.
    """
    skippy_node: SkippyNode
    ether_node: EtherNode

    docker_images: Set = set()

    @property
    def name(self):
        return self.ether_node.name

    @property
    def arch(self):
        return self.ether_node.arch


class SimulationTimeoutError(BaseException):
    pass


class Environment(simpy.Environment):
    cluster: ClusterContext

    def __init__(self, initial_time=0):
        super().__init__(initial_time)
        self.faas = None
        self.topology = None
        self.benchmark = None
        self.cluster = None
        self.registry = None
        self.metrics = None
        self.node_states = dict()

    def get_node_state(self, name: str) -> Optional[NodeState]:
        if name in self.node_states:
            return self.node_states[name]

        node = self.topology.find_node(name)
        if node:
            node_state = NodeState()
            node_state.ether_node = node
            self.node_states[name] = node_state
            return node_state

        return None


def timeout_listener(env, started, max_time, interval=1):
    while True:
        yield env.timeout(interval)

        if (time.time() - started) > max_time:
            raise SimulationTimeoutError()
