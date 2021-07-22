import abc
from simpy.core import Environment

from ext.jjnp21.scalers.lb_scaler import LoadBalancerScaler


class LoadBalancerScalerFactory(abc.ABC):
    @abc.abstractmethod
    def create(self, env: Environment) -> LoadBalancerScaler: ...



