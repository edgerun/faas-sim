import abc
from simpy.core import Environment

from ext.jjnp21.schedulers.lb_schedulers.lb_scheduler import LoadBalancerScheduler


class LoadBalancerSchedulerFactory(abc.ABC):
    @abc.abstractmethod
    def create(self, env: Environment) -> LoadBalancerScheduler: ...

    