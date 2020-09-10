import time
from typing import Set

import simpy
from ether.core import Node as EtherNode

from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Node as SkippyNode


class Node:
    skippy_node: SkippyNode
    ether_node: EtherNode

    docker_images: Set = {}

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
        self.metrics = Metrics()


def timeout_listener(env, started, max_time, interval=1):
    while True:
        yield env.timeout(interval)

        if (time.time() - started) > max_time:
            raise SimulationTimeoutError()
