import abc
import statistics
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
    def __init__(self, env: Environment, lbs: List[LoadBalancerReplica]):
        self.lbs = lbs
        self.env = env
        self.client_map: Dict[int, LocalizedLoadBalancer] = dict()

    def reset(self, lbs: List[LoadBalancerReplica]):
        self.lbs = lbs
        self.client_map = dict()

    def _distance(self, client: Node, lb: LocalizedLoadBalancer):
        samples = []
        for _ in range(10):
            samples.append(self.env.topology.latency(client, lb.ether_node))
        return statistics.mean(samples)

    def _find_closest_lb(self, client: Node) -> LocalizedLoadBalancer:
        min_dist_lb = None
        min_dist = 100000000000
        for lb in self.lbs:
            if not isinstance(lb.load_balancer, LocalizedLoadBalancer):
                continue
            if min_dist_lb is None or self._distance(client, lb.load_balancer) < min_dist:
                min_dist = self._distance(client, lb.load_balancer)
                min_dist_lb = lb.load_balancer
        return min_dist_lb

    def get_closest_lb(self, client_node: Node):
        if self.client_map.get(id(client_node)) is None:
            self.client_map[id(client_node)] = self._find_closest_lb(client_node)
        return self.client_map[id(client_node)]


class LocalizedLRT(LeastResponseTimeLoadBalancer, LocalizedLoadBalancer):

    def next_replica(self, request) -> FunctionReplica:
        request.load_balancer = self
        return super().next_replica(request)


class LocalizedRR(RoundRobinLoadBalancer, LocalizedLoadBalancer):

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        request.load_balancer = self
        return super().next_replica(request)