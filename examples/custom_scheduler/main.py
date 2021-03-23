import logging
import random

from skippy.core.clustercontext import ClusterContext
from skippy.core.model import SchedulingResult, Pod

import examples.basic.main as basic
from sim.core import Environment
from sim.faassim import Simulation

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(basic.example_topology(), basic.ExampleBenchmark())

    # override the scheduler factory
    sim.create_scheduler = CustomScheduler.create

    # run the simulation
    sim.run()


class CustomScheduler:
    """
    Example scheduler implementation that picks a node at random.
    """

    def __init__(self, cluster: ClusterContext):
        self.cluster = cluster

    def schedule(self, pod: Pod) -> SchedulingResult:
        """
        Schedule selects a node for a Pod (in Kubernetes language). Our system assumes that Kubernetes or a similar
        platform is used as underlying runtime for the FaaS system.
        """

        # get all available nodes in the cluster from the cluster context
        nodes = self.cluster.list_nodes()

        # pick a node at random
        node = random.choice(nodes)

        logger.info("selected node %s for pod %s from total of %d nodes", node.name, pod.name, len(nodes))

        # the last two arguments of SchedulingResult (feasible_nodes, needed_images) are not needed
        return SchedulingResult(node, len(nodes), list())

    @staticmethod
    def create(env: Environment):
        """
        Factory method that is injected into the Simulation
        """
        logger.info('creating CustomScheduler')
        # the ClusterContext holds the cluster state (concepts from skippy)
        return CustomScheduler(env.cluster)


if __name__ == '__main__':
    main()
