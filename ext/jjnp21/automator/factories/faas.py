from ext.jjnp21.localized_lb_system import LocalizedLoadBalancerFaasSystem
from ext.jjnp21.localized_osmotic_lb_system import OsmoticLoadBalancerCapableFaasSystem
from sim.core import Environment
from sim.faas import FaasSystem


class FaaSFactory:
    constructor_kwargs = {}
    def set_constructor_args(self, **kwargs):
        self.constructor_kwargs = kwargs

    def create(self, env: Environment) -> FaasSystem:
        raise Exception('Do not use this class directly. Use an actual implementation.')


class LocalizedLoadBalancerFaaSFactory(FaaSFactory):
    def create(self, env: Environment) -> FaasSystem:
        return LocalizedLoadBalancerFaasSystem(env, **self.constructor_kwargs)


class OsmoticLoadBalancerCapableFaasSystemFactory(FaaSFactory):
    def create(self, env: Environment) -> FaasSystem:
        return OsmoticLoadBalancerCapableFaasSystem(env, **self.constructor_kwargs)
