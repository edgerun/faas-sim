from typing import Dict, List

from ext.jjnp21.core import LoadBalancerDeployment
from ext.jjnp21.load_balancers.lrt import LeastResponseTimeLoadBalancer
from sim.core import Environment
from sim.faas import LoadBalancer, FunctionReplica


class LRTLoadBalancerDeployment(LoadBalancerDeployment):
    def __init__(self, lrt_window: float = 15, weight_update_frequency: float = 15):
        self.lrt_window = lrt_window
        self.weight_update_frequency = weight_update_frequency
        # todo add super constructor call with other parametrization information

    def create_load_balancer(self, env: Environment, replicas: Dict[str, List[FunctionReplica]]) -> LoadBalancer:
        return LeastResponseTimeLoadBalancer(env, replicas, lrt_window=self.lrt_window,
                                             weight_update_frequency=self.weight_update_frequency)
