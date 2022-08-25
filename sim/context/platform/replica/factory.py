import uuid
from typing import Dict

from faas.context import NodeService, FunctionDeploymentService, FunctionReplicaFactory, InMemoryFunctionReplicaService
from faas.system import FunctionContainer, FunctionReplicaState

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.context.platform.node.model import SimFunctionNode
from sim.context.platform.replica.model import SimFunctionReplica
from sim.context.platform.replica.service import SimFunctionReplicaService


class KubernetesFunctionReplicaFactory(FunctionReplicaFactory[SimFunctionDeployment, SimFunctionReplica]):

    def create_replica(self, labels: Dict[str, str], fn_container: FunctionContainer,
                       fn_deployment: SimFunctionDeployment) -> SimFunctionReplica:
        image = fn_container.fn_image.image.split('/')[1].split(':')[0]
        uid = uuid.uuid4()
        replica_id = f'{image}-{uid}'

        return SimFunctionReplica(
            replica_id,
            labels,
            fn_deployment,
            fn_container,
            None,
            FunctionReplicaState.CONCEIVED
        )


def create_replica_service(node_service: NodeService[SimFunctionNode],
                           deployment_service: FunctionDeploymentService[SimFunctionDeployment]):
    in_memory_function_service = InMemoryFunctionReplicaService[SimFunctionReplica](node_service, deployment_service,
                                                                                    KubernetesFunctionReplicaFactory())
    return SimFunctionReplicaService(in_memory_function_service)
