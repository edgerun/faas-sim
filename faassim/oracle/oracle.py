from typing import Tuple
import glob
import pandas as pd
from numpy.random.mtrand import normal

from core.clustercontext import ClusterContext
from core.model import Pod, Node


class Oracle:
    """Abstract class for placement oracle functions."""
    def estimate(self, context: ClusterContext, pod: Pod, node: Node) -> Tuple[str, str]:
        raise NotImplementedError


class PlacementTimeOracle(Oracle):
    def __init__(self):
        placement_csvs = glob.glob('sim/oracle/pod_placement_*.csv')
        dfs = [pd.read_csv(filename) for filename in placement_csvs]
        self.df = pd.concat(dfs)
        # Failed deployments can be neglected as they must not be scheduled by a correct scheduler anyways
        # TODO transform the dataframe s.t. we can read the median placement time for an image on a
        #  specific node type with a given bandwidth and if the image is present or not
        #  node types:
        #  bandwidth:
        #

    def estimate(self, context: ClusterContext, pod: Pod, node: Node) -> Tuple[str, str]:
        # TODO implement placement time estimation for the pod being placed on the node
        return 'placement_time', str(normal(loc=1337))


class ExecutionTimeOracle(Oracle):
    def estimate(self, context: ClusterContext, pod: Pod, node: Node) -> Tuple[str, str]:
        # TODO implement execution time estimation for the pod being executed on the node
        return 'execution_time', str(normal(loc=1337))