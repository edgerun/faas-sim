from typing import Optional, List, Set

import numpy as np
from faas.system import FunctionNode
from sklearn.base import RegressorMixin

from sim.context.platform.request.service import RequestService
from sim.core import Environment
from sim.degradation import create_degradation_model_input, DegradationTrace
from sim.oracle.oracle import ResourceOracle


class SimFunctionNode(FunctionNode):
    """
        Holds simulation specific runtime knowledge about a node. For example, what docker images it has already pulled.
        """
    docker_images: Set
    performance_degradation: Optional[RegressorMixin]

    def __init__(self, fn_node: FunctionNode) -> None:
        super().__init__(fn_node.name, fn_node.arch, fn_node.cpus, fn_node.ram, fn_node.netspeed, fn_node.labels,
                         fn_node.allocatable,
                         fn_node.cluster, fn_node.state)
        self.ether_node = None
        self.skippy_node = None
        self.docker_images = set()
        self.performance_degradation = None
        self.cache = {}

    def estimate_degradation(self, env: Environment,resource_oracle: ResourceOracle,
                             start_ts: int, end_ts: int) -> float:
        if self.performance_degradation is not None:
            rounded_start = round(start_ts, 1)
            rounded_end = round(end_ts, 1)
            get = self.cache.get((rounded_start, rounded_end), None)
            if get is not None:
                return get

            calls = self.get_calls_in_timeframe(env, self.name, start_ts, end_ts)
            x = create_degradation_model_input(calls, start_ts, end_ts, self.name,
                                               self.ether_node.capacity.memory, resource_oracle)

            if len(x) == 0:
                # in case no other calls happened
                return 0
            x = np.array(x).reshape((1, -1))
            y = self.performance_degradation.predict(x)[0]
            self.cache[(rounded_start, rounded_end)] = y
            return y
        return 0

    def get_calls_in_timeframe(self, env: Environment, node: str, start_ts: float, end_ts: float) -> List[DegradationTrace]:
        calls = []
        request_service: RequestService = env.context.request_service
        for inflight_request in request_service.get_inflight_request(node):
            trace = DegradationTrace(inflight_request.replica, inflight_request.start, end_ts)
            end_ts = request_service.get_end_ts(inflight_request.request_id)
            if inflight_request.start <= start_ts:
                if end_ts is None or end_ts > start_ts:
                    calls.append(inflight_request)

            else:
                # all calls that started afterwards but before the end are relevant
                if inflight_request.start < end_ts:
                    calls.append(trace)

        return calls
