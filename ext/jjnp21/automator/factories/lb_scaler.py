import abc

from simpy.core import Environment

# from ext.jjnp21.core import LoadBalancerDeployment
from ext.jjnp21.scalers.fraction_lb_scaler import FractionScaler
from ext.jjnp21.scalers.lb_scaler import LoadBalancerScaler
from ext.jjnp21.scalers.manual_set_lb_scaler import ManualSetScaler
from ext.jjnp21.scalers.osmotic_lb_scaler import OsmoticLoadBalancerScaler


class LoadBalancerScalerFactory(abc.ABC):
    kwargs = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def create(self, ld: 'LoadBalancerDeployment', env: Environment) -> LoadBalancerScaler: ...


class FractionLoadBalancerScalerFactory(LoadBalancerScalerFactory):

    def __init__(self, target_fraction: float = 0.1):
        super().__init__(target_fraction=target_fraction)

    def create(self, ld: 'LoadBalancerDeployment', env: Environment) -> LoadBalancerScaler:
        return FractionScaler(ld, env, **self.kwargs)


class ManualSetLoadBalancerScalerFactory(LoadBalancerScalerFactory):
    def __init__(self, target_count: int = 1):
        super().__init__(target_count=target_count)

    def create(self, ld: 'LoadBalancerDeployment', env: Environment) -> LoadBalancerScaler:
        return ManualSetScaler(ld, env, **self.kwargs)


class OsmoticLoadBalancerScalerFactory(LoadBalancerScalerFactory):
    def __init__(self, pressure_threshold: float, hysteresis: float):
        super().__init__(pressure_threshold=pressure_threshold, hysteresis=hysteresis)

    def create(self, ld: 'LoadBalancerDeployment', env: Environment) -> LoadBalancerScaler:
        return OsmoticLoadBalancerScaler(ld, env, **self.kwargs)



