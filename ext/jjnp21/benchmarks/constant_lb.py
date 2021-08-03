from typing import List

from ext.jjnp21.core import LoadBalancerDeployment
from ext.jjnp21.deployments.lrt_lb import LRTLoadBalancerDeployment, RRLoadBalancerDeployment
from ext.jjnp21.localized_lb_system import LoadBalancerCapableFaasSystem
from ext.raith21.benchmark.constant import ConstantBenchmark
from ext.jjnp21.misc.images import all_lb_images
from sim.benchmark import BenchmarkBase
from sim.core import Environment

class LoadBalancerConstant(ConstantBenchmark):
    def __init__(self, profile: str, duration: int, rps=200, model_folder=None, lb_profile: str = 'LRT'):
        self.lb_deployments: List[LoadBalancerDeployment] = self.create_lb_deployments_for_profile(lb_profile)
        super().__init__(profile, duration, rps=rps, model_folder=model_folder)
        self.images.extend(all_lb_images)

    def setup(self, env: Environment):
        super().setup(env)


    def setup_lb_deployments(self):
        pass

    def run(self, env: Environment):
        if not isinstance(env.faas, LoadBalancerCapableFaasSystem):
            raise Exception('Tried to use a LB-enabled benchmark with a faas system that is not load-balancer enabled.'
                            'Please use a load-balancer capable faas-system instead.')
        # deploy lb-replicas
        for ld in self.lb_deployments:
            yield from env.faas.deploy_lb(ld)
        # wait until one replica of each load balancer deployment is running
        for ld in self.lb_deployments:
            yield env.process(env.faas.poll_available_lb_replica(ld.name))
        # call the supercall run() method to also start the function replicas. note that you MUST use yield from
        # figuring that out took an hour out of my life...
        yield from super().run(env)


    def create_lb_deployments_for_profile(self, profile: str) -> List[LoadBalancerDeployment]:
        if profile == 'LRT':
            return self.__create_lrt_deployments()
        elif profile == 'RR':
            return self.__create_rr_deployments()
        else:
            raise Exception(f'Invalid LB-profile for benchmark: {profile}')

    def __create_lrt_deployments(self) -> List[LoadBalancerDeployment]:
        return [LRTLoadBalancerDeployment()]

    def __create_rr_deployments(self) -> List[LoadBalancerDeployment]:
        raise [RRLoadBalancerDeployment()]



