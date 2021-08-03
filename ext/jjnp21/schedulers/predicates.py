from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Pod, Node
from skippy.core.predicates import Predicate

from sim.topology import supports_central_load_balancer


class CanHostCentralLoadBalancer(Predicate):
    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        return node.labels.get(supports_central_load_balancer) is not None


class NoLoadBalancerRunningYet(Predicate):
    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        # Returns true if the node does not have a pod with a load-balancer image running as of yet
        # Assumes that each load balancer image name contains the string 'load_balancer'
        for p in node.pods:
            for c in p.spec.containers:
                if 'load_balancer' in c.image:
                    return False
        return True