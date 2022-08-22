from faas.util.constant import hostname_label
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Pod, Node
from skippy.core.predicates import Predicate



class PodHostEqualsNode(Predicate):

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        host = pod.spec.labels.get(hostname_label)
        if host is None:
            return True
        else:
            return node.name == host

