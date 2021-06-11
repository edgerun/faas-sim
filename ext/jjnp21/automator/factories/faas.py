from ext.jjnp21.localized_lb_system import LocalizedLoadBalancerFaasSystem
from sim.core import Environment
from sim.faas import FaasSystem


class FaaSFactory:
    def create(self, env: Environment) -> FaasSystem:
        raise Exception('Do not use this class directly. Use an actual implementation.')


class LocalizedLoadBalancerFaaSFactory(FaaSFactory):
    def create(self, env: Environment) -> FaasSystem:
        return LocalizedLoadBalancerFaasSystem(env, scale_by_queue_requests_per_replica=True)
