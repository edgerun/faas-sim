from typing import Dict, List
import random
from ether.core import Node
from sim.core import Environment
from sim.faas import FunctionReplica, LoadBalancer, FunctionRequest
from sim.faas import RoundRobinLoadBalancer


class LocalizedRoundRobinLoadBalancer(RoundRobinLoadBalancer):
    def __init__(self, env: Environment, node: Node, replicas: Dict[str, List[FunctionReplica]]) -> None:
        self.node = node
        super().__init__(env, replicas)

    def next_replica(self, request) -> FunctionReplica:
        request.load_balancer = self
        return super().next_replica(request)


# TODO: Clean up this wrapper
class LocalizedRRLBWrapper(LoadBalancer):
    def __init__(self, load_balancers: List[LocalizedRoundRobinLoadBalancer]):
        self.load_balancers = load_balancers

    def next_replica(self, request: FunctionRequest) -> FunctionReplica:
        return self.choose_load_balancer().next_replica(request)

    def choose_load_balancer(self) -> LocalizedRoundRobinLoadBalancer:
        return random.choice(self.load_balancers)
