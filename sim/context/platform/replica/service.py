from typing import Union, List, Optional, Dict, Callable

from faas.context import FunctionReplicaService, InMemoryFunctionReplicaService
from faas.context.observer.api import Observer
from faas.system import FunctionReplicaState, FunctionReplica

from sim.context.platform.replica.model import SimFunctionReplica


class SimFunctionReplicaService(FunctionReplicaService[SimFunctionReplica]):

    def __init__(self, replica_service: InMemoryFunctionReplicaService[SimFunctionReplica]):
        self.replica_service = replica_service
        self.observers: List[Observer] = []

    def find_by_predicate(self, predicate: Callable[[SimFunctionReplica], bool], running: bool = True,
                          state: FunctionReplicaState = None) -> \
            List[SimFunctionReplica]:
        return self.replica_service.find_by_predicate(predicate, running, state)

    def get_function_replicas(self) -> List[SimFunctionReplica]:
        return self.replica_service.get_function_replicas()

    def get_function_replicas_of_deployment(self, deployment_name, running: bool = True,
                                            state: FunctionReplicaState = None) -> \
            List[SimFunctionReplica]:
        return self.replica_service.get_function_replicas_of_deployment(deployment_name, running, state)

    def find_function_replicas_with_labels(self, labels: Dict[str, str] = None, node_labels=None, running: bool = True,
                                           state: FunctionReplicaState = None) -> List[
        SimFunctionReplica]:
        return self.replica_service.find_function_replicas_with_labels(labels, node_labels, running, state)

    def get_function_replica_by_id(self, replica_id: str) -> Optional[SimFunctionReplica]:
        return self.replica_service.get_function_replica_by_id(replica_id)

    def get_function_replicas_on_node(self, node_name: str) -> List[SimFunctionReplica]:
        return self.replica_service.get_function_replicas_on_node(node_name)

    def shutdown_function_replica(self, replica_id: str):
        self.replica_service.shutdown_function_replica(replica_id)

    def add_function_replica(self, replica: SimFunctionReplica) -> SimFunctionReplica:
        return self.replica_service.add_function_replica(replica)

    def delete_function_replica(self, replica_id: str):
        self.replica_service.delete_function_replica(replica_id)

    def scale_down(self, function_name: str, remove: Union[int, List[SimFunctionReplica]]) -> List[SimFunctionReplica]:
        removed = self.replica_service.scale_down(function_name, remove)
        return removed

    def scale_up(self, function_name: str, add: Union[int, List[SimFunctionReplica]]) -> List[SimFunctionReplica]:
        added = self.replica_service.scale_up(function_name, add)
        return added

    def register(self, observer: Observer):
        self.replica_service.register(observer)

    def set_state(self, replica: FunctionReplica, state: FunctionReplicaState):
        self.replica_service.set_state(replica, state)

