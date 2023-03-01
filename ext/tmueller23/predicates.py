from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Pod, Node
from skippy.core.predicates import Predicate


hostname_label = 'skippy.core.hostname'

class HostnamePredicate(Predicate):
    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        pod_label = pod.spec.labels.get(hostname_label)
        if not pod_label:
            return True

        return node.name == pod_label
