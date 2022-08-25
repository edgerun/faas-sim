from typing import Optional, List, Set

import numpy as np
from faas.system import FunctionNode
from sklearn.base import RegressorMixin

from sim.degradation import create_degradation_model_input
from sim.oracle.oracle import ResourceOracle


class SimFunctionNode(FunctionNode):
    """
        Holds simulation specific runtime knowledge about a node. For example, what docker images it has already pulled.
        """
    docker_images: Set
    current_requests: Set
    all_requests: List[any]
    performance_degradation: Optional[RegressorMixin]

    def __init__(self, fn_node: FunctionNode) -> None:
        super().__init__(fn_node.name, fn_node.arch, fn_node.cpus, fn_node.ram, fn_node.netspeed, fn_node.labels,
                         fn_node.allocatable,
                         fn_node.cluster, fn_node.state)
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
