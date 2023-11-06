from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from faas.system.core import FaasSystem
from faas.util.rwlock import ReadWriteLock

from sim.core import Environment
from sim.faas import SimFunctionReplica


@dataclass
class ResourceUsage:
    ts: float
    value: float


class ResourceUtilization:
    # TODO: delete obsolete (i.e., via a time frame argument) data
    __resources: Dict[str, Dict[int, ResourceUsage]]
    __add: Dict[int, float]
    __remove: Dict[int, float]

    def __init__(self, env: Environment):
        self.idx = defaultdict(int)
        self.env = env
        self.__resources = defaultdict(dict)
        self.__add = {}
        self.__remove = {}
        self.lock = ReadWriteLock()

    def put_resource(self, resource: str, value: float) -> int:
        with self.lock.lock.gen_wlock():
            idx = self.idx[resource]
            self.idx[resource] += 1
        ts = self.env.now
        self.__resources[resource][idx] = ResourceUsage(ts, value)
        self.__add[idx] = ts
        return idx

    def remove_resource(self, key: int):
        self.__remove[key] = self.env.now

    def list_resources(self) -> List[str]:
        return list(self.__resources.keys())

    def copy(self) -> 'ResourceUtilization':
        with self.lock.lock.gen_rlock():
            util = ResourceUtilization(self.env)
            util.__resources = deepcopy(self.__resources)
            util.__add = deepcopy(self.__add)
            util.__remove = deepcopy(self.__remove)
            return util

    def get_resource(self, resource: str, start: float, end: float, time_step: float = 1) -> Optional[pd.DataFrame]:
        if start < 0:
            start = 0
        keys = []
        for key, ts in self.__add.items():
            if ts >= start and ts <= end:
                keys.append(key)
            if ts < start and self.__remove.get(key) is None:
                keys.append(key)
        resource_usages = self.__resources.get(resource)
        if resource_usages is None:
            return None
        step = start
        usage = defaultdict(list)
        while step < end:
            tmp_end = step + time_step
            value = 0
            for key in keys:
                if self.__add.get(key) is None or self.__resources[resource].get(key) is None:
                    continue
                if self.__remove.get(key) is not None and self.__remove[key] < step:
                    # this is past resource and does not concern us
                    continue
                elif self.__add[key] < step and self.__remove.get(key) is None:
                    # the resource started before and is still ongoing
                    value += self.__resources[resource][key].value
                elif self.__add[key] < step and self.__remove[key] > tmp_end:
                    # the resource started before and ends after our horizon
                    value += self.__resources[resource][key].value
                elif self.__add[key] >= step and self.__remove.get(key) is None:
                    # the resource started after our start but is still ongoing
                    duration = tmp_end - self.__add[key]
                    ratio = duration / time_step
                    value += self.__resources[resource][key].value * ratio
                elif self.__add[key] >= step and self.__remove[key] > tmp_end:
                    # the resource started after our start but ends after our horizon
                    duration = tmp_end - self.__add[key]
                    ratio = duration / time_step
                    value += self.__resources[resource][key].value * ratio
                elif self.__add[key] >= step and self.__remove[key] <= tmp_end:
                    duration = self.__remove[key] - self.__add[key]
                    ratio = duration / time_step
                    value += self.__resources[resource][key].value * ratio

            if value != 0:
                usage['value'].append(value)
                usage['ts'].append(step)
                usage['resource'].append(resource)
            step += time_step
        return pd.DataFrame(data=usage)

    def get_average_resource(self, resource, start: float, end: float, time_step: float = 1) -> Optional[float]:
        resource = self.get_resource(resource, start, end, time_step)
        if resource is None or len(resource) == 0:
            return None
        return resource['value'].mean()

    def is_empty(self) -> bool:
        return len(self.__resources) == 0


class NodeResourceUtilization:
    # key is replica_id,  uniqueness allows for running same FunctionContainer multiple times on node
    __resources: Dict[str, ResourceUtilization]

    # associates the replica_id with its SimFunctionReplica
    __replicas: Dict[str, SimFunctionReplica]

    def __init__(self, env: Environment, resources: List[str] = None):
        self.env = env
        self.resources = resources
        if resources is None:
            self.resources = []
        self.__resources = {}
        self.__replicas = {}

    def put_resource(self, replica: SimFunctionReplica, resource: str, value: float) -> int:
        return self.get_resource_utilization(replica).put_resource(resource, value)

    def remove_resource(self, replica: SimFunctionReplica, resource_index: int):
        self.get_resource_utilization(replica).remove_resource(resource_index)

    def get_resource_utilization(self, replica: SimFunctionReplica) -> ResourceUtilization:
        name = replica.replica_id
        util = self.__resources.get(name)
        if util is None:
            self.__resources[name] = ResourceUtilization(self.env)
            self.__replicas[name] = replica
            return self.__resources[name]
        else:
            return util

    def get_average_resource_utilization(self, replica: SimFunctionReplica, start: float, end: float,
                                         time_step: float = 1) -> Optional[pd.DataFrame]:
        name = replica.replica_id
        util = self.__resources.get(name)
        if util is None:
            self.__resources[name] = ResourceUtilization(self.env)
            self.__replicas[name] = replica
            return None
        else:
            data = defaultdict(list)
            for resource in util.list_resources():
                usage = util.get_average_resource(resource, start, end, time_step)
                if usage != None:
                    data['ts'].append(end)
                    data['value'].append(usage)
                    data['resource'].append(resource)
            return pd.DataFrame(data=data)

    def list_resource_utilization(self) -> List[Tuple[SimFunctionReplica, ResourceUtilization]]:
        functions = []
        for replica_id, utilization in self.__resources.items():
            replica = self.__replicas.get(replica_id)
            functions.append((replica, utilization))
        return functions

    def total_utilization(self, start: float, end: float, timestep: float) -> Optional[pd.DataFrame]:
        dfs = []
        for replica, resource_utilization in self.list_resource_utilization():
            for resource in resource_utilization.list_resources():
                usage_frame = resource_utilization.get_resource(resource, start, end, timestep)
                if usage_frame is not None and len(usage_frame) > 0:
                    usage_frame['replica_id'] = replica.replica_id
                    dfs.append(usage_frame)
        if len(dfs) > 0:
            df = pd.concat(dfs)
            df = df.groupby(['resource', 'ts']).sum().reset_index()
            return df
        else:
            return None

    def copy(self) -> 'NodeResourceUtilization':
        resources = {}
        for replica_id, util in self.__resources.items():
            resources[replica_id] = util.copy()

        replicas = self.__replicas.copy()

        util = NodeResourceUtilization(self.env, self.resources)
        util.__resources = resources
        util.__replicas = replicas
        return util


# TODO: include observer that fires in put and remove events. This can enable a metric logging implementation that always
# has all data and can be used for post-experiment analysis.
class ResourceState:
    node_resource_utilizations: Dict[str, NodeResourceUtilization]

    def __init__(self, env: Environment, resources: List[str] = None):
        self.env = env
        self.resources = resources
        if resources is None:
            self.resources = []
        self.node_resource_utilizations = {}

    def put_resource(self, function_replica: 'SimFunctionReplica', resource: str, value: float) -> int:
        node_name = function_replica.node.name
        node_resources = self.get_node_resource_utilization(node_name)
        return node_resources.put_resource(function_replica, resource, value)

    def remove_resource(self, replica: 'SimFunctionReplica', resource_index: int):
        node_name = replica.node.name
        self.get_node_resource_utilization(node_name).remove_resource(replica, resource_index)

    def get_resource_utilization(self, replica: 'SimFunctionReplica', start: float, end: float,
                                 time_step: float = 1) -> 'ResourceUtilization':
        node_name = replica.node.name
        util = self.get_node_resource_utilization(node_name).get_resource_utilization(replica)
        for resource in self.resources:
            if util.get_resource(resource, start, end, time_step) is None:
                util.put_resource(resource, 0)
        return util

    def get_average_resource_utilization(self, replica: 'SimFunctionReplica', start: float,
                                         end: float, time_step: float = 1) -> pd.DataFrame:
        node_name = replica.node.name
        util = self.get_node_resource_utilization(node_name).get_average_resource_utilization(replica, start, end,
                                                                                              time_step)
        return util

    def get_average_node_resource_utilization(self, node_name: str, start: float, end: float) -> Optional[
        NodeResourceUtilization]:
        pass

    def list_resource_utilization(self, node_name: str) -> List[Tuple['SimFunctionReplica', 'ResourceUtilization']]:
        return self.get_node_resource_utilization(node_name).list_resource_utilization()

    def get_node_resource_utilization(self, node_name: str) -> Optional[NodeResourceUtilization]:
        node_resources = self.node_resource_utilizations.get(node_name)
        if node_resources is None:
            self.node_resource_utilizations[node_name] = NodeResourceUtilization(self.env, self.resources)
            node_resources = self.node_resource_utilizations[node_name]
        return node_resources


@dataclass
class ReplicaResourceWindow:
    replica: SimFunctionReplica
    resources: pd.DataFrame
    time: int


@dataclass
class NodeResourceWindow:
    node: str
    resources: pd.DataFrame
    time: int


class ResourceMonitor:
    """Simpy process - continuously collects resource data puts it into the TelemetryService"""

    def __init__(self, env: Environment, reconcile_interval: float, logging=True):
        self.env = env
        self.reconcile_interval = reconcile_interval
        self.logging = logging

    def run(self):
        faas: FaasSystem = self.env.faas
        while True:
            yield self.env.timeout(self.reconcile_interval)
            now = self.env.now
            start = now - 1
            end = now
            state: ResourceState = self.env.resource_state
            replica_service = self.env.context.replica_service
            telemetry_service = self.env.context.telemetry_service
            deployment_service = self.env.context.deployment_service
            for deployment in deployment_service.get_deployments():
                replicas: List[SimFunctionReplica] = replica_service.get_function_replicas_of_deployment(
                    deployment.name, running=True)
                for replica in replicas:
                    utilization: pd.DataFrame = state.get_average_resource_utilization(replica, start, end)
                    if utilization is None or len(utilization) == 0:
                        continue
                    # TODO extract logging into own process
                    if self.logging:
                        self.env.metrics.log_function_resource_utilization(replica, utilization)
                    telemetry_service.put_replica_resource_utilization(
                        ReplicaResourceWindow(replica, utilization, now))
            for node in self.env.topology.get_nodes():
                resource_utilization: pd.DataFrame = state.get_node_resource_utilization(node.name).total_utilization(
                    start, end, 1)
                if resource_utilization is not None:
                    telemetry_service.put_node_resource_utilization(
                        NodeResourceWindow(node.name, resource_utilization, now))
                if self.logging:
                    if resource_utilization is not None:
                        self.env.metrics.log_resource_utilization(node.name, node.capacity, resource_utilization)
