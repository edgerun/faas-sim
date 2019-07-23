from typing import Tuple, NamedTuple
import glob
import pandas as pd
from ast import literal_eval as make_tuple

from core.clustercontext import ClusterContext
from core.model import Pod, SchedulingResult
from core.utils import parse_size_string, normalize_image_name

Bandwidth = NamedTuple('Bandwidth', [('mbit', int), ('delay', int), ('deviation', int)])


class Oracle:
    def __init__(self, filename):
        csvs = glob.glob(filename)
        dfs = [pd.read_csv(filename) for filename in csvs]
        df = pd.concat(dfs)
        # Filter failed ones (training on pi)
        df = df.loc[df['status'].isin(['passed'])]
        # Transform the bandwidth to Bytes/s
        df['bandwidth'] = df['bandwidth'].apply(lambda x: eval(x))
        # Assume 10 GBit for no limit
        df['bandwidth'] = df['bandwidth'].apply(lambda x: 1.25e+8 if x is None else parse_size_string(f'{x.mbit}M') / 8)
        # Transform the hostname to only contain the type (cloud, tegra, pi)
        df['host'] = df['host'].apply(lambda x: make_tuple(x)[0][:-1])
        self.dataset = df

    """Abstract class for placement oracle functions."""
    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        raise NotImplementedError


class PlacementTimeOracle(Oracle):
    def __init__(self):
        super(PlacementTimeOracle, self).__init__('sim/oracle/pod_placement_*.csv')
        # Perform the group by to calc the median time for
        # each host x with bandwidth y, image z and image present or not
        self.grouped_dataset = self.dataset[['host', 'bandwidth', 'image', 'image_present', 'duration']]\
            .groupby(['host', 'bandwidth', 'image', 'image_present'])
        self.durations = self.grouped_dataset.median()

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'placement_time', None
        host = scheduling_result.suggested_host.name
        host_type = host[host.rindex('_')+1:]
        # For the placement time the bandwidth to the registry is necessary
        bandwidth = context.get_bandwidth_graph()[host]['registry']
        placement_time = 0
        for container in pod.spec.containers:
            image = container.image
            image_present =  normalize_image_name(image) in context.images_on_nodes[host]
            placement_time += self.durations.query(f'host == "{host_type}" and bandwidth == {bandwidth} and '
                                                  f'image == "{image}" and '
                                                  f'image_present == {image_present}')['duration'].values[0]
        # return 'placement_time', str(normal(placement_time, placement_time * 0.1))
        return 'placement_time', str(placement_time)


class ExecutionTimeOracle(Oracle):
    def __init__(self):
        super(ExecutionTimeOracle, self).__init__('sim/oracle/exec_time*.csv')
        # Perform the group by to calc the median time for
        # each host x with bandwidth y and image z
        self.grouped_dataset = self.dataset[['host', 'bandwidth', 'image', 'duration']]\
            .groupby(['host', 'bandwidth', 'image'])
        self.durations = self.grouped_dataset.median()

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'execution_time', None
        host = scheduling_result.suggested_host.name
        host_type = host[host.rindex('_')+1:]
        # For the execution time the bandwidth to the next storage node is necessary
        bandwidth = context.get_bandwidth_graph()[host][context.get_next_storage_node(scheduling_result.suggested_host)]
        execution_time = 0
        for container in pod.spec.containers:
            image = container.image
            if image == 'alexrashed/ml-wf-3-serve:0.33':
                execution_time = None
                break
            execution_time += self.durations.query(f'host == "{host_type}" and '
                                                   f'bandwidth == {bandwidth} and '
                                                   f'image == "{image}"')['duration'].values[0]
        # return 'execution_time', 'None' if execution_time is None else str(normal(execution_time, execution_time * 0.1))
        return 'execution_time', str(execution_time)