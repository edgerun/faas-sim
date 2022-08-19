from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sim.core import Environment
from sim.faas import SimFunctionReplica
from faas.system.core import FaasSystem


class ResourceUtilization:
    __resources: Dict[str, float]

    def __init__(self):
        self.__resources = {}

    def put_resource(self, resource: str, value: float):
        if self.__resources.get(resource) is None:
            self.__resources[resource] = 0
        self.__resources[resource] += value

    def remove_resource(self, resource: str, value: float):
        if self.__resources.get(resource) is None:
            self.__resources[resource] = 0
        self.__resources[resource] -= value

    def list_resources(self) -> Dict[str, float]:
        return deepcopy(self.__resources)

    def copy(self) -> 'ResourceUtilization':
        util = ResourceUtilization()
        util.__resources = self.list_resources()
        return util

    def get_resource(self, resource) -> Optional[float]:
        return self.__resources.get(resource)

    def is_empty(self) -> bool:
        return len(self.__resources) == 0


class NodeResourceUtilization:
    # key is pod-name,   uniqueness allows for running same FunctionContainer multiple times on node
    __resources: Dict[str, ResourceUtilization]

    # associates the pod-name with its SimFunctionReplica
    __replicas: Dict[str, SimFunctionReplica]

    def __init__(self):
        self.__resources = {}
        self.__replicas = {}

    def put_resource(self, replica: SimFunctionReplica, resource: str, value: float):
        self.get_resource_utilization(replica).put_resource(resource, value)

    def remove_resource(self, replica: SimFunctionReplica, resource: str, value: float):
        self.get_resource_utilization(replica).remove_resource(resource, value)

    def get_resource_utilization(self, replica: SimFunctionReplica) -> ResourceUtilization:
        name = replica.pod.name
        util = self.__resources.get(name)
        if util is None:
            self.__resources[name] = ResourceUtilization()
            self.__replicas[name] = replica
            return self.__resources[name]
        else:
            return util

    def list_resource_utilization(self) -> List[Tuple[SimFunctionReplica, ResourceUtilization]]:
        functions = []
        for pod_name, utilization in self.__resources.items():
            replica = self.__replicas.get(pod_name)
            functions.append((replica, utilization))
        return functions

    @property
    def total_utilization(self) -> ResourceUtilization:
        total = ResourceUtilization()
        for _, resource_utilization in self.list_resource_utilization():
            for resource, value in resource_utilization.list_resources().items():
                total.put_resource(resource, value)
        return total


class ResourceState:
    node_resource_utilizations: Dict[str, NodeResourceUtilization]

    def __init__(self):
        self.node_resource_utilizations = {}

    def put_resource(self, function_replica: SimFunctionReplica, resource: str, value: float):
        node_name = function_replica.node.name
        node_resources = self.get_node_resource_utilization(node_name)
        node_resources.put_resource(function_replica, resource, value)

    def remove_resource(self, replica: 'SimFunctionReplica', resource: str, value: float):
        node_name = replica.node.name
        self.get_node_resource_utilization(node_name).remove_resource(replica, resource, value)

    def get_resource_utilization(self, replica: 'SimFunctionReplica') -> 'ResourceUtilization':
        node_name = replica.node.name
        return self.get_node_resource_utilization(node_name).get_resource_utilization(replica)

    def list_resource_utilization(self, node_name: str) -> List[Tuple['SimFunctionReplica', 'ResourceUtilization']]:
        return self.get_node_resource_utilization(node_name).list_resource_utilization()

    def get_node_resource_utilization(self, node_name: str) -> Optional[NodeResourceUtilization]:
        node_resources = self.node_resource_utilizations.get(node_name)
        if node_resources is None:
            self.node_resource_utilizations[node_name] = NodeResourceUtilization()
            node_resources = self.node_resource_utilizations[node_name]
        return node_resources


@dataclass
class ResourceWindow:
    replica: SimFunctionReplica
    resources: Dict[str, float]
    time: float


class MetricsServer:
    """
    contains methods to obtain metrics - offers query functions for resources (functionreplica)

    stores time-series data in data structure (i.e. list)

    """

    def __init__(self):
        # TODO this will inevitably leak memory
        self._windows = defaultdict(lambda: defaultdict(list))

    # TODO make dynamic -> read key-values from replica/pod
    def put(self, window: ResourceWindow):
        node = window.replica.node.name
        pod = window.replica.pod.name

        self._windows[node][pod].append(window)

    def get_average_cpu_utilization(self, fn_replica: SimFunctionReplica, window_start: float,
                                    window_end: float) -> float:
        utilization = self.get_average_resource_utilization(fn_replica, 'cpu', window_start, window_end)
        millis = fn_replica.node.capacity.cpu_millis
        return utilization / millis

    def get_average_resource_utilization(self, fn_replica: SimFunctionReplica, resource: str, window_start: float,
                                         window_end: float) -> float:
        node = fn_replica.node.name
        pod = fn_replica.pod.name
        windows: List[ResourceWindow] = self._windows.get(node, {}).get(pod, [])
        if len(windows) == 0:
            return 0
        average_windows = []

        for window in reversed(windows):
            if window.time <= window_end:
                if window.time < window_start:
                    break
                average_windows.append(window)
        # slicing never throws IndexError
        return np.mean(list(map(lambda l: l.resources[resource], average_windows)))


class ResourceMonitor:
    """Simpy process - continuously collects resource data"""

    def __init__(self, env: Environment, reconcile_interval: int, logging=True):
        self.env = env
        self.reconcile_interval = reconcile_interval
        self.metric_server: MetricsServer = env.metrics_server
        self.logging = logging

    def run(self):
        faas: FaasSystem = self.env.faas
        while True:
            yield self.env.timeout(self.reconcile_interval)
            now = self.env.now
            for deployment in faas.get_deployments():
                replicas: List[SimFunctionReplica] = faas.get_replicas(deployment.name, True, )
                for replica in replicas:
                    utilization = self.env.resource_state.get_resource_utilization(replica)
                    if utilization.is_empty():
                        continue
                    # TODO extract logging into own process
                    if self.logging:
                        self.env.metrics.log_function_resource_utilization(replica, utilization)
                    self.metric_server.put(
                        ResourceWindow(replica, utilization.list_resources(), now))
