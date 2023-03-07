from typing import List

from faas.util.constant import controller_role_label
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Pod, Node
from skippy.core.predicates import Predicate


class CheckNodeLabelPresencePred(Predicate):
    def __init__(self, labels: List[str]) -> None:
        super().__init__()
        self.labels = labels

        self._passes_predicate = self.has_labels

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        return self._passes_predicate(pod, node)

    def has_labels(self, pod: Pod, node: Node) -> bool:
        for label in self.labels:
            if label not in pod.spec.labels:
                return True
            if label not in node.labels:
                return False

        return True

class ExclusivePred(Predicate):

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        return len(node.pods) == 0

class LoadBalancerPred(Predicate):

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        if pod.spec.labels.get(controller_role_label) == 'true' and node.labels.get(controller_role_label) == 'true':
            return True
        if pod.spec.labels.get(controller_role_label) == 'true'  and node.labels.get(controller_role_label) is None:
            return False
        if pod.spec.labels.get(controller_role_label) is None  and node.labels.get(controller_role_label) == 'true':
            return False
        return True
