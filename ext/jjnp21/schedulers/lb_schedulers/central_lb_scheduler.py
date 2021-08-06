import logging
from typing import List

from skippy.core.predicates import Predicate

from ext.jjnp21.schedulers.lb_schedulers.lb_scheduler import LoadBalancerScheduler
from ext.jjnp21.schedulers.lb_schedulers.random_lb_scheduler import RandomLoadBalancerScheduler
from ext.jjnp21.schedulers.predicates import NoLoadBalancerRunningYet, CanHostCentralLoadBalancer
from sim.core import Environment

logger = logging.getLogger(__name__)


class CentralLoadBalancerScheduler(RandomLoadBalancerScheduler):

    @staticmethod
    def create(env: Environment, predicates: List[Predicate]) -> LoadBalancerScheduler:
        """
        Factory method that is injected into the Simulation
        """
        logger.info('creating "everywhere" load balancer scheduler')
        changed_predicates = predicates.copy()
        changed_predicates.append(NoLoadBalancerRunningYet())
        changed_predicates.append(CanHostCentralLoadBalancer())
        return CentralLoadBalancerScheduler(env.cluster, predicates)
