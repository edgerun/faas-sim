import logging
from collections import defaultdict
from typing import Set, Optional, Any, Generator, Callable, List, Dict

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
        self.buffer_size = 0
        self.buffer_limit = 50
        self.cache = {}

    def estimate_degradation(self, start_ts: int, end_ts: int) -> float:
        if self.performance_degradation is not None:
            rounded_start = round(start_ts, 1)
            rounded_end = round(end_ts, 1)
            get = self.cache.get((rounded_start, rounded_end), None)
            if get is not None:
                return get

            x = create_degradation_model_input(self, start_ts, end_ts)

            if len(x) == 0:
                # in case no other calls happened
                return 0
            x = np.array(x).reshape((1, -1))
            y = self.performance_degradation.predict(x)[0]
            self.cache[(rounded_start, rounded_end)] = y
            return y
        return 0

    def clean_up(self):
        if self.buffer_size >= self.buffer_limit:
            remove_candidates = [x for x in self.all_requests if x.end is not None]
            not_remove = set()
            for req in self.all_requests:
                if req.end is not None:
                    continue
                for past_request in remove_candidates:
                    if req.start < past_request.end:
                        not_remove.add(past_request)
            for req in not_remove:
                remove_candidates.remove(req)
            for req in remove_candidates:
                self.all_requests.remove(req)
            self.buffer_size = self.buffer_size - len(remove_candidates)
        self.buffer_size += 1

    def set_end(self, request_id, end):
        for call in self.all_requests:
            if call.request_id == request_id:
                call.end = end

        self.clean_up()

    @property
    def name(self):
        return self.ether_node.name

    @property
    def arch(self):
        return self.ether_node.arch

    @property
    def capacity(self):
        return self.ether_node.capacity

    def get_calls_in_timeframe(self, start_ts, end_ts):
        calls = []
        for call in self.all_requests:
            if call.start <= start_ts:
                # add only calls that are either not finished or have finished afterwards
                if call.end is None or call.end > start_ts:
                    calls.append(call)
            else:
                # all calls that started afterwards but before the end are relevant
                if call.start < end_ts:
                    calls.append(call)
        return calls


def create_degradation_model_input(node_state: NodeState, start_ts: int, end_ts: int) -> np.ndarray:
    # input of model is an array with 34 elements
    # in general, the input is based on the resource usages that occurred during the function execution
    # for each trace (instance) from the target service following metrics
    # for each resource, the millis of usage per image get recorded
    # i.e. if image A has 2 requests, during the execution and each call had 200 millis CPU, the input for this
    # image is the sum of the millis = 400.
    # after having summed up all usages per image, the input is formed by calculating the following measures:
    # mean, std dev, min, max, 25 percentile, 50 percentile and 75 percentile
    # this translates to the following indices
    # ['cpu', 'gpu', 'blkio', 'net']
    # 0 - 6: cpu mean, cpu std dev,...
    # 7 - 13: gpu mean, gpu std dev...
    # 14 - 20: blkio mean, blkio std dev...
    # 21 - 27: net mean, net std dev...
    # 28: number of running containers that have executed at least one call
    # 29: sum of all cpu millis
    # 30: sum of all gpu millis
    # 31: sum of all blkio rate ! not scaled
    # 32: sum of all net rate ! not scaled
    # 33: mean ram percentage over complete experiment
    resources_types = ['cpu', 'gpu', 'blkio', 'net']

    calls = node_state.get_calls_in_timeframe(start_ts, end_ts)
    if len(calls) == 0:
        return np.array([])
    ram = 0
    seen_pods = set()
    resources = defaultdict(lambda: defaultdict(list))
    for call in calls:
        function = call.replica.function
        pod_name = call.replica.pod.name

        call_resources = function.get_resources_for_node(node_state.name)
        for resource_type in resources_types:
            resources[pod_name][resource_type].append(call_resources[resource_type])
        if len(call_resources) == 0:
            logging.debug(f'Function {function.name} has no resources for node {node_state.name}')
            continue

        if pod_name not in seen_pods:
            ram += call.replica.pod.spec.containers[0].resources.requests['memory']
            seen_pods.add(pod_name)
        last_start = start_ts if start_ts >= call.start else call.start

        if call.end is not None:
            first_end = end_ts if end_ts <= call.end else call.end
        else:
            first_end = end_ts

        overlap = first_end - last_start

        for resource in resources_types:
            resources[pod_name][resource].append(overlap * call_resources[resource])

    sums = defaultdict(list)
    for resource_type in resources_types:
        for pod_name, resources_of_pod in resources.items():
            resource_sum = np.sum(resources_of_pod[resource_type])
            sums[resource_type].append(resource_sum)

    # make input for model
    # the values get converted to a fixed length array, i.e.: descriptive statistics
    # of the resources of all faas containers
    # skip the first element, it's only the count of containers
    input = []
    for resource in resources_types:
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
    for resource in resources_types:
        input.append(np.sum(sums[resource]))

    # add ram_rate in percentage too
    input.append(ram / node_state.capacity.memory)

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
        self.metrics_server = None
        self.background_processes: List[Callable[[Environment], Generator[simpy.events.Event, Any, Any]]] = []
        self.degradation_models: Dict[str, Optional[RegressorMixin]] = {}

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

        degradation_model = self.degradation_models.get(name, None)
        if degradation_model is not None:
            node_state.performance_degradation = degradation_model

        self.node_states[name] = node_state
        return node_state


def timeout_listener(env, started, max_time, interval=1):
    while True:
        yield env.timeout(interval)

        if (time.time() - started) > max_time:
            raise SimulationTimeoutError()
