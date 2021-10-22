import abc

from simpy.core import Environment

from ext.jjnp21.schedulers.lb_schedulers.central_lb_scheduler import CentralLoadBalancerScheduler
from ext.jjnp21.schedulers.lb_schedulers.everywhere_lb_scheduler import EverywhereLoadBalancerScheduler
from ext.jjnp21.schedulers.lb_schedulers.lb_scheduler import LoadBalancerScheduler
from ext.jjnp21.schedulers.lb_schedulers.osmotic_lb_scheduler import OsmoticLoadBalancerScheduler


class LoadBalancerSchedulerFactory(abc.ABC):
    @abc.abstractmethod
    def create(self, env: Environment) -> LoadBalancerScheduler: ...


class EverywhereLoadBalancerSchedulerFactory(LoadBalancerSchedulerFactory):
    def create(self, env: Environment) -> LoadBalancerScheduler:
        return EverywhereLoadBalancerScheduler.create(env, [])


class CentralLoadBalancerSchedulerFactory(LoadBalancerSchedulerFactory):
    def create(self, env: Environment) -> LoadBalancerScheduler:
        return CentralLoadBalancerScheduler.create(env, [])


class OsmoticLoadBalancerSchedulerFactory(LoadBalancerSchedulerFactory):
    def create(self, env: Environment) -> LoadBalancerScheduler:
        return OsmoticLoadBalancerScheduler.create(env, [])
