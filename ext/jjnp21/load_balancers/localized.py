import abc
import statistics
from collections import defaultdict
from typing import List, Dict

from ether.core import Node

from ext.jjnp21.core import LoadBalancerReplica
from ext.jjnp21.load_balancers.lrt import LeastResponseTimeLoadBalancer
from sim.core import Environment
from sim.faas import LoadBalancer, FunctionReplica, RoundRobinLoadBalancer, FunctionRequest


class LocalizedLoadBalancer(LoadBalancer, abc.ABC):
    ether_node: Node = None

    def set_node(self, ether_node: Node):
        self.ether_node = ether_node


class ClosestLoadBalancerFinder:
    def __init__(self, env: Environment, replicas: List[LoadBalancerReplica]):
        self.replicas = replicas
        self.env = env
        self.client_map: Dict[Node, (LoadBalancerReplica, float)] = dict()
        self.count = 0
        self.ctr = defaultdict(lambda: 0)

    def add(self, new_replica: LoadBalancerReplica):
        self.replicas.append(new_replica)
        for node, (_, dist) in self.client_map.items():
            new_dist = self._distance(node, new_replica)
            if new_dist < dist:
                self.client_map[node] = (new_replica, new_dist)

    def remove(self, del_replica: LoadBalancerReplica):
        deletion_key = None
        for node, (replica, _) in self.client_map.items():
            if replica == del_replica:
                deletion_key = node
        if deletion_key is not None:
            del self.client_map[deletion_key]
        self.replicas.remove(del_replica)

    def _distance(self, client: Node, replica: LoadBalancerReplica):
        samples = []
        for _ in range(10):
            samples.append(self.env.topology.latency(client, replica.node.ether_node))
        return statistics.mean(samples)

    def _find_closest_lb(self, client: Node) -> (LoadBalancerReplica, float):
        min_dist_lb = None
        min_dist = 100000000000
        for replica in self.replicas:
            if not isinstance(replica, LoadBalancerReplica):
                continue
            if min_dist_lb is None or self._distance(client, replica) < min_dist:
                min_dist = self._distance(client, replica)
                min_dist_lb = replica
        return min_dist_lb, min_dist

    def get_closest_lb(self, client_node: Node) -> LoadBalancerReplica:
        self.count += 1
        if self.client_map.get(client_node) is None:
            self.client_map[client_node] = self._find_closest_lb(client_node)
        self.ctr[self.client_map[client_node][0]] += 1
        return self.client_map[client_node][0]


class LocalizedLRT(LeastResponseTimeLoadBalancer, LocalizedLoadBalancer):

    def next_replica(self, request) -> FunctionReplica:
        request.load_balancer = self
        return super().next_replica(request)


class LocalizedRR(RoundRobinLoadBalancer, LocalizedLoadBalancer):

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        request.load_balancer = self
        return super().next_replica(request)