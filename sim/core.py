import time
from typing import Set, Optional, Any, Generator, Callable, List, Dict

import numpy as np
import simpy
from ether.core import Node as EtherNode, Capacity
from sklearn.base import RegressorMixin

from .degradation import create_degradation_model_input
from .oracle.oracle import ResourceOracle

Node = EtherNode


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

    def estimate_degradation(self, resource_oracle: ResourceOracle,
                             start_ts: int, end_ts: int) -> float:
        if self.performance_degradation is not None:
            rounded_start = round(start_ts, 1)
            rounded_end = round(end_ts, 1)
            get = self.cache.get((rounded_start, rounded_end), None)
            if get is not None:
                return get

            calls = self.get_calls_in_timeframe(start_ts, end_ts)
            x = create_degradation_model_input(calls, start_ts, end_ts, self.name,
                                               self.capacity.memory, resource_oracle)

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
    def capacity(self) -> Capacity:
        return self.ether_node.capacity

    def get_calls_in_timeframe(self, start_ts, end_ts) -> List:
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


class SimulationTimeoutError(BaseException):
    pass


class Environment(simpy.Environment):
    cluster: 'SimulationClusterContext'
    faas: 'FaasSystem'

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
        self.resource_state = None
        self.resource_monitor = None
        self.flow_factory = None
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
