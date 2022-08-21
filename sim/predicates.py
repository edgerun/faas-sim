from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Pod, Node
from skippy.core.predicates import Predicate

host_label = 'faassim.edgerun.io/hostname'


class PodHostEqualsNode(Predicate):

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        host = pod.spec.labels.get(host_label)
        if host is None:
            return True
        else:
            return node.name == host

