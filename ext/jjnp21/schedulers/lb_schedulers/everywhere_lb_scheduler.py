from itertools import islice, cycle
from typing import List

import random
import logging

from skippy.core.predicates import Predicate
from skippy.core.utils import normalize_image_name
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import SchedulingResult, Pod, Node

from ext.jjnp21.schedulers.function_schedulers.random_scheduler import RandomScheduler
from ext.jjnp21.schedulers.lb_schedulers.lb_scheduler import LoadBalancerScheduler
from ext.jjnp21.schedulers.lb_schedulers.random_lb_scheduler import RandomLoadBalancerScheduler
from ext.jjnp21.schedulers.predicates import NoLoadBalancerRunningYet
from sim.core import Environment

logger = logging.getLogger(__name__)


class EverywhereLoadBalancerScheduler(RandomLoadBalancerScheduler):
    pass

    @staticmethod
    def create(env: Environment, predicates: List[Predicate]) -> LoadBalancerScheduler:
        """
        Factory method that is injected into the Simulation
        """
        logger.info('creating "everywhere" load balancer scheduler')
        changed_predicates = predicates.copy()
        changed_predicates.append(NoLoadBalancerRunningYet())
        return EverywhereLoadBalancerScheduler(env.cluster, changed_predicates)
