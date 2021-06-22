import random
from typing import Dict, List
from ether.core import Node
from .lrt import LeastResponseTimeLoadBalancer
from sim.core import Environment
from sim.faas import FunctionReplica, LoadBalancer, FunctionRequest
import statistics


class LocalizedLeastResponseTimeLoadBalancer(LeastResponseTimeLoadBalancer):
    def __init__(self, env: Environment, node: Node, replicas: Dict[str, List[FunctionReplica]],
                 lrt_window: float = 60, weight_update_frequency: float = 10) -> None:
        self.node = node
        super().__init__(env, replicas, lrt_window, weight_update_frequency)

    def next_replica(self, request) -> FunctionReplica:
        request.load_balancer = self
        return super().next_replica(request)


class LocalizedLRTLBWrapper(LoadBalancer):
    def __init__(self, load_balancers: List[LocalizedLeastResponseTimeLoadBalancer], env: Environment):
        self.load_balancers = load_balancers
        self.env = env
        # Attention! This client map assumed that clients and LBs are static and do not change over time.
        # If they do this must be adapted
        self.client_map: Dict[int, LocalizedLeastResponseTimeLoadBalancer] = dict()

    def _distance_between_lb_and_client(self, client: Node, lb: LocalizedLeastResponseTimeLoadBalancer):
        # sample 10 times
        samples = []
        for i in range(0, 10):
            samples.append(self.env.topology.latency(client, lb.node))
        return statistics.median(samples)

    def _find_closest_lb_for_client(self, client_node: Node) -> LocalizedLeastResponseTimeLoadBalancer:
        min_distance_lb = None
        min_distance = 1000000000000000
        for lb in self.load_balancers:
            if min_distance_lb is None or self._distance_between_lb_and_client(client_node, lb) < min_distance:
                min_distance = self._distance_between_lb_and_client(client_node, lb)
                min_distance_lb = lb
        return min_distance_lb

    def _choose_lb_for_client(self, client_node: Node) -> LocalizedLeastResponseTimeLoadBalancer:
        if self.client_map.get(id(client_node)) is None:
        # if not isinstance(self.client_map.get(id(client_node)), LoadBalancer):
            self.client_map[id(client_node)] = self._find_closest_lb_for_client(client_node)
        return self.client_map[id(client_node)]
        # return self._find_closest_lb_for_client(client_node)

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        if hasattr(request, 'client_node'):
            return self._choose_lb_for_client(request.client_node).next_replica(request)
        return self.choose_random_load_balancer().next_replica(request)

    def choose_random_load_balancer(self) -> LocalizedLeastResponseTimeLoadBalancer:
        return random.choice(self.load_balancers)

