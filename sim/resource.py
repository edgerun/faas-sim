from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from faas.system.core import FaasSystem

from sim.core import Environment
from sim.faas import SimFunctionReplica


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
    # key is replica_id,  uniqueness allows for running same FunctionContainer multiple times on node
    __resources: Dict[str, ResourceUtilization]

    # associates the replica_id with its SimFunctionReplica
    __replicas: Dict[str, SimFunctionReplica]

    def __init__(self, resources: List[str] = None):
        self.resources = resources
        if resources is None:
            self.resources = []
        self.__resources = {}
        self.__replicas = {}

    def put_resource(self, replica: SimFunctionReplica, resource: str, value: float):
        self.get_resource_utilization(replica).put_resource(resource, value)

    def remove_resource(self, replica: SimFunctionReplica, resource: str, value: float):
        self.get_resource_utilization(replica).remove_resource(resource, value)

    def get_resource_utilization(self, replica: SimFunctionReplica) -> ResourceUtilization:
        name = replica.replica_id
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
        if total.is_empty():
            for resource in self.resources:
                total.put_resource(resource, 0)
        return total

    def copy(self) -> 'NodeResourceUtilization':
        resources = {}
        for replica_id, util in self.__resources.items():
            resources[replica_id] = util.copy()

        replicas = self.__replicas.copy()

        util = NodeResourceUtilization(self.resources)
        util.__resources = resources
        util.__replicas = replicas
        return util


class ResourceState:
    node_resource_utilizations: Dict[str, NodeResourceUtilization]

    def __init__(self, resources: List[str] = None):
        self.resources = resources
        if resources is None:
            self.resources = []
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
        util = self.get_node_resource_utilization(node_name).get_resource_utilization(replica)
        for resource in self.resources:
            if util.get_resource(resource) is None:
                util.put_resource(resource, 0)
        return util

    def list_resource_utilization(self, node_name: str) -> List[Tuple['SimFunctionReplica', 'ResourceUtilization']]:
        return self.get_node_resource_utilization(node_name).list_resource_utilization()

    def get_node_resource_utilization(self, node_name: str) -> Optional[NodeResourceUtilization]:
        node_resources = self.node_resource_utilizations.get(node_name)
        if node_resources is None:
            self.node_resource_utilizations[node_name] = NodeResourceUtilization(self.resources)
            node_resources = self.node_resource_utilizations[node_name]
        return node_resources


@dataclass
class ReplicaResourceWindow:
    replica: SimFunctionReplica
    resources: 'ResourceUtilization'
    time: int


@dataclass
class NodeResourceWindow:
    node: str
    resources: NodeResourceUtilization
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
            state: ResourceState = self.env.resource_state
            replica_service = self.env.context.replica_service
            telemetry_service = self.env.context.telemetry_service
            deployment_service = self.env.context.deployment_service
            for deployment in deployment_service.get_deployments():
                replicas: List[SimFunctionReplica] = replica_service.get_function_replicas_of_deployment(
                    deployment.name, running=True)
                for replica in replicas:
                    utilization = state.get_resource_utilization(replica)
                    if utilization.is_empty():
                        continue
                    # TODO extract logging into own process
                    if self.logging:
                        self.env.metrics.log_function_resource_utilization(replica, utilization)
                    telemetry_service.put_replica_resource_utilization(
                        ReplicaResourceWindow(replica, utilization.copy(), now))
            for node in self.env.topology.get_nodes():
                resource_utilization: NodeResourceUtilization = state.get_node_resource_utilization(node.name)
                telemetry_service.put_node_resource_utilization(
                    NodeResourceWindow(node.name, resource_utilization.copy(), now))
                if self.logging:
                    total_utilization = resource_utilization.total_utilization
                    self.env.metrics.log_resource_utilization(node.name, node.capacity, total_utilization)
