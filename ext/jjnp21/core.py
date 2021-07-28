from sim.core import Environment
from sim.faas import FunctionDeployment, FunctionReplica, LoadBalancer


class LoadBalancerDeployment(FunctionDeployment):
    def create_load_balancer(self, env: Environment) -> LoadBalancer:
        pass


class LoadBalancerReplica(FunctionReplica):
    load_balancer: LoadBalancer = None