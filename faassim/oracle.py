from typing import Tuple

from numpy.random.mtrand import normal

from core.clustercontext import ClusterContext
from core.model import Pod, Node


class Oracle:
    """Abstract class for placement oracle functions."""
    def estimate(self, context: ClusterContext, pod: Pod, node: Node) -> Tuple[str, str]:
        raise NotImplementedError


class PlacementTimeOracle(Oracle):
    def estimate(self, context: ClusterContext, pod: Pod, node: Node) -> Tuple[str, str]:
        # TODO implement placement time estimation for the pod being placed on the node
        return 'placement_time', str(normal(loc=1337))


class ExecutionTimeOracle(Oracle):
    def estimate(self, context: ClusterContext, pod: Pod, node: Node) -> Tuple[str, str]:
        # TODO implement execution time estimation for the pod being executed on the node
        return 'execution_time', str(normal(loc=1337))