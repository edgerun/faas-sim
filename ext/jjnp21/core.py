from typing import Dict, List, Generator, Tuple

from ext.jjnp21.localized_lb_system import LoadBalancerCapableFaasSystem
from sim.benchmark import BenchmarkBase
from sim.core import Environment
from sim.faas import FunctionDeployment, FunctionReplica, LoadBalancer


class LoadBalancerDeployment(FunctionDeployment):
    def create_load_balancer(self, env: Environment, replicas: Dict[str, List[FunctionReplica]]) -> LoadBalancer:
        raise Exception('This is just a base class. Use an actual implementation of this class instead')


class LoadBalancerReplica(FunctionReplica):
    load_balancer: LoadBalancer = None


class LBBenchmark(BenchmarkBase):
    def __init__(self, images: List[Tuple[str, str, str]], deployments: List[FunctionDeployment],
                 arrival_profiles: Dict[str, Generator], lb_deployments: List[LoadBalancerDeployment]):
        self.lb_deployments = lb_deployments
        self.lb_deployments_per_name = self.__create_lb_deployments_per_name()
        super().__init__(images, deployments, arrival_profiles)

    def __create_lb_deployments_per_name(self):
        lb_deployments_per_name = {}
        for ld in self.lb_deployments:
            lb_deployments_per_name[ld.name] = ld
        return lb_deployments_per_name

    def run(self, env: Environment):
        if not isinstance(env.faas, LoadBalancerCapableFaasSystem):
            raise Exception(
                'Used a load-balancer enabled benchmark with a faas-system, that does not support load-balancing. '
                'Please use a load balancer capable faas-system instead.')
        faas: LoadBalancerCapableFaasSystem = env.faas
        # deploy all load-balancers
        for ld in self.lb_deployments:
            yield from faas.deploy_lb(ld)
        # wait until one replica of each load balancer deployment is running
        for ld in self.lb_deployments:
            yield env.process(faas.poll_available_faas_replica(ld.name))
        super().run(env)
