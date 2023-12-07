import uuid
from copy import deepcopy
from typing import Dict

from faas.context import NodeService, FunctionDeploymentService, FunctionReplicaFactory, InMemoryFunctionReplicaService
from faas.system import FunctionContainer, FunctionReplicaState

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.context.platform.node.model import SimFunctionNode
from sim.context.platform.replica.model import SimFunctionReplica
from sim.context.platform.replica.service import SimFunctionReplicaService
from sim.core import Environment
from sim.skippy import create_function_pod


class SimFunctionReplicaFactory(FunctionReplicaFactory[SimFunctionDeployment, SimFunctionReplica]):

    def create_replica(self, labels: Dict[str, str], fn_container: FunctionContainer,
                       fn_deployment: SimFunctionDeployment) -> SimFunctionReplica:
        replica_id = fn_deployment.name + '-' + str(uuid.uuid4())[12:].replace('-', '')

        replica = SimFunctionReplica(
            replica_id,
            deepcopy(labels),
            fn_deployment,
            fn_container,
            None,
            FunctionReplicaState.CONCEIVED
        )
        replica.pod = self.create_pod(fn_deployment, fn_container)

        return replica

    def create_pod(self, fd: SimFunctionDeployment, fn: FunctionContainer):
        return create_function_pod(fd, fn)

def create_replica_service(node_service: NodeService[SimFunctionNode],
                           deployment_service: FunctionDeploymentService[SimFunctionDeployment], env: Environment):
    in_memory_function_service = InMemoryFunctionReplicaService[SimFunctionReplica](node_service, deployment_service,
                                                                                    SimFunctionReplicaFactory())
    return SimFunctionReplicaService(in_memory_function_service, env)
