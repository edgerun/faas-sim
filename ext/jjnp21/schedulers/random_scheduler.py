from itertools import islice, cycle
from typing import List

import random
import logging

from skippy.core.predicates import Predicate
from skippy.core.utils import normalize_image_name
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import SchedulingResult, Pod, Node

from sim.core import Environment

logger = logging.getLogger(__name__)


class RandomScheduler:
    """
    Picks a node (almost) at random. It checks which nodes match the predicates, and then chooses one based on that.
    """

    def __init__(self, cluster: ClusterContext, predicates: List[Predicate]):
        self.cluster = cluster
        self.predicates = predicates

    def schedule(self, pod: Pod) -> SchedulingResult:
        """
        Schedule selects a node for a Pod (in Kubernetes language). Our system assumes that Kubernetes or a similar
        platform is used as underlying runtime for the FaaS system.
        """

        # get all available nodes in the cluster from the cluster context
        nodes = self.cluster.list_nodes()

        filtered = list(filter(lambda node: self.passes_predicates(pod, node), nodes))

        # pick a node at random
        suggested_host = None
        if len(filtered) > 0:
            suggested_host = random.choice(filtered)
        needed_images = None

        if suggested_host is not None:
            # Add a list of images needed to pull to the result (before manipulating the state with #place_pod_on_node
            needed_images = []
            host_images = self.cluster.images_on_nodes[suggested_host.name]
            for container in pod.spec.containers:
                if normalize_image_name(container.image) not in host_images:
                    needed_images.append(normalize_image_name(container.image))

            self.cluster.place_pod_on_node(pod, suggested_host)

            logger.info("selected node %s for pod %s from total of %d nodes", suggested_host.name, pod.name, len(filtered))

        # the last two arguments of SchedulingResult (feasible_nodes, needed_images) are not needed
        return SchedulingResult(suggested_host, len(filtered), needed_images)

    # taken directly from the skippy implementation
    def passes_predicates(self, pod: Pod, node: Node) -> bool:
        # Conjunction over all node predicate checks
        return all(self.__passes_and_logs_predicate(predicate, self.cluster, pod, node)
                   for predicate in self.predicates)

    # noinspection PyMethodMayBeStatic
    def __passes_and_logs_predicate(self, predicate: Predicate, context: ClusterContext, pod: Pod, node: Node):
        result = predicate.passes_predicate(context, pod, node)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Pod {pod.name} / Node {node.name} / {type(predicate).__name__}: '
                         f'{"Passed" if result else "Failed"}')
        return result

    @staticmethod
    def create(env: Environment, predicates: List[Predicate]):
        """
        Factory method that is injected into the Simulation
        """
        logger.info('creating RandomScheduler')
        # the ClusterContext holds the cluster state (concepts from skippy)
        return RandomScheduler(env.cluster, predicates)
