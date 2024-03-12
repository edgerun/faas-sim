from faas.system.core import FunctionReplica, FunctionContainer, \
    FunctionReplicaState
from skippy.core.model import Pod

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.context.platform.node.model import SimFunctionNode


class SimFunctionReplica(FunctionReplica):
    """
    A function replica is an instance of a function running on a specific node.
    """
    function: SimFunctionDeployment
    container: FunctionContainer
    node: SimFunctionNode
    pod: Pod
    state: FunctionReplicaState = FunctionReplicaState.CONCEIVED

    simulator: 'FunctionSimulator' = None

    @property
    def fn_name(self):
        return self.function.name

    @property
    def image(self):
        return self.container.image

    def __str__(self):
        return f'{self.pod.name} {self.function.name} {self.state.value}'
