from collections import defaultdict
from typing import Set, Optional, Any, Generator, Callable, List

import joblib
import numpy as np
import simpy
from ether.core import Node as EtherNode
from sklearn.base import RegressorMixin

Node = EtherNode

import time


class NodeState:
    """
    Holds simulation specific runtime knowledge about a node. For example, what docker images it has already pulled.
    """
    docker_images: Set
    current_requests: Set
    all_requests: List[any]
    performance_degradation: Optional[RegressorMixin]

    def __init__(self) -> None:
        super().__init__()
        self.ether_node = None
        self.skippy_node = None
        self.docker_images = set()
        self.current_requests = set()
        self.all_requests = []
        self.performance_degradation = None

    def estimate_degradation(self, start_ts: int, end_ts: int) -> float:
        if self.performance_degradation is None:
            self.performance_degradation = joblib.load(
                f'./data/{self.ether_node.name[:self.ether_node.name.rindex("_")]}.sav')
        x = create_input(self, start_ts, end_ts)
        return self.performance_degradation.predict([x])[0]

    def set_end(self, request_id, end):
        for call in self.all_requests:
            if call.request_id == request_id:
                call.end = end

    @property
    def name(self):
        return self.ether_node.name

    @property
    def arch(self):
        return self.ether_node.arch

    @property
    def capacity(self):
        return self.ether_node.capacity


def create_input(node_state: NodeState, start_ts: int, end_ts: int) -> np.ndarray:
    """
    input of model is an array with 25 elements
    in general, the input is based on the resource usages that occurred during the function execution
    for each resource, the millis of usage per image get recorded
    i.e. if image A has 2 requests, during the execution and each call had 200 millis CPU, the input for this
    image is the sum of the millis = 400.
    after having summed up all usages per image, the input is formed by calculating the following measures:
    mean, std dev, min, max, 25 percentile, 50 percentile and 75 percentile
    this translates to the following indices
    0 - 6: cpu mean, cpu std dev,...
    7 - 13: gpu mean, gpu std dev...
    14 - 20: io mean, io std dev...
    21: number of running containers that have executed at least one call
    22: sum of all cpu millis
    23: sum of all gpu millis
    24: sum of all io millis
    """
    calls = []
    for call in node_state.all_requests:
        if call.start <= start_ts:
            # add only calls that are either not finished or have finished afterwards
            if call.end is None or call.end > start_ts:
                calls.append(call)
        else:
            # all calls that started afterwards but before the end are relevant
            if call.start < end_ts:
                calls.append(call)

    resources = defaultdict(lambda: defaultdict(list))
    for call in calls:
        cpu_usage = float(call.replica.function.labels.get('cpu', '0'))
        io_usage = float(call.replica.function.labels.get('io', '0'))
        gpu_usage = float(call.replica.function.labels.get('gpu', '0'))

        last_start = start_ts if start_ts >= call.start else call.start

        if call.end is not None:
            first_end = end_ts if end_ts <= call.end else call.end
        else:
            first_end = end_ts

        overlap = first_end - last_start

        resources[call.replica.pod.name]['cpu'].append(overlap * cpu_usage)
        resources[call.replica.pod.name]['gpu'].append(overlap * gpu_usage)
        resources[call.replica.pod.name]['io'].append(overlap * io_usage)

    sums = defaultdict(list)
    for pod in resources.keys():
        cpu_sum = np.sum(resources[pod]['cpu'])
        gpu_sum = np.sum(resources[pod]['gpu'])
        io_sum = np.sum(resources[pod]['io'])
        sums['cpu'].append(cpu_sum)
        sums['gpu'].append(gpu_sum)
        sums['io'].append(io_sum)

    # make input for model
    # the values get converted to a fixed length array, i.e.: descriptive statistics
    # of the resources of all faas containers
    # skip the first element, it's only the count of containers
    input = []
    for resource in ['cpu', 'gpu', 'io']:
        mean = np.mean(sums[resource])
        std = np.std(sums[resource])
        amin = np.min(sums[resource])
        amax = np.max(sums[resource])
        p_25 = np.percentile(sums[resource], q=0.25)
        p_50 = np.percentile(sums[resource], q=0.5)
        p_75 = np.percentile(sums[resource], q=0.75)
        for value in [mean, std, amin, amax, p_25, p_50, p_75]:
            # in case of only one container the std will be np.nan
            if np.isnan(value):
                input.append(0)
            else:
                input.append(value)

    # add number of containers
    input.append(len(sums['cpu']))

    # add total sums resources
    for resource in ['cpu', 'gpu', 'io']:
        input.append(np.sum(sums[resource]))

    return np.array(input)


class SimulationTimeoutError(BaseException):
    pass


class Environment(simpy.Environment):
    cluster: 'SimulationClusterContext'

    def __init__(self, initial_time=0):
        super().__init__(initial_time)
        self.faas = None
        self.simulator_factory = None
        self.topology = None
        self.storage_index = None
        self.benchmark = None
        self.cluster = None
        self.container_registry = None
        self.metrics = None
        self.scheduler = None
        self.node_states = dict()
        self.background_processes: List[Callable[[Environment], Generator[simpy.events.Event, Any, Any]]] = []

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
