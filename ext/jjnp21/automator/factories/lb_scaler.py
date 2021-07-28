import abc
from simpy.core import Environment

from ext.jjnp21.core import LoadBalancerDeployment
from ext.jjnp21.scalers.lb_scaler import LoadBalancerScaler


class LoadBalancerScalerFactory(abc.ABC):
    kwargs = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def create(self, ld: LoadBalancerDeployment, env: Environment) -> LoadBalancerScaler: ...



