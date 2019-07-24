from typing import Tuple, NamedTuple
import glob
import pandas as pd
from ast import literal_eval as make_tuple

from core.clustercontext import ClusterContext
from core.model import Pod, SchedulingResult, Node, ImageState
from core.utils import parse_size_string, normalize_image_name

Bandwidth = NamedTuple('Bandwidth', [('mbit', int), ('delay', int), ('deviation', int)])


class Oracle:
    """Abstract class for placement oracle functions."""
    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        raise NotImplementedError


class EmpiricalOracle:
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


class PlacementTimeOracle(EmpiricalOracle):
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
            image_present = normalize_image_name(image) in context.images_on_nodes[host]
            placement_time += self.durations.query(f'host == "{host_type}" and bandwidth == {bandwidth} and '
                                                  f'image == "{image}" and '
                                                  f'image_present == {image_present}')['duration'].values[0]
        # return 'placement_time', str(normal(placement_time, placement_time * 0.1))
        return 'placement_time', str(placement_time)


class ExecutionTimeOracle(EmpiricalOracle):
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
                # TODO do a proper estimation something here!
                # execution_time = None
                execution_time = 2
                break
            execution_time += self.durations.query(f'host == "{host_type}" and '
                                                   f'bandwidth == {bandwidth} and '
                                                   f'image == "{image}"')['duration'].values[0]
        # return 'execution_time', 'None' if execution_time is None else str(normal(execution_time, execution_time * 0.1))
        return 'execution_time', str(execution_time)


class BandwidthUsageOracle(Oracle):
    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'bandwidth_usage', None

        # Calculate the image pull bandwidth
        bandwidth_usage = 0
        node = scheduling_result.suggested_host
        if pod.spec.containers is not Node:
            for container in pod.spec.containers:
                image_name = normalize_image_name(container.image)
                if image_name not in context.images_on_nodes[node.name]:
                    try:
                        image_state: ImageState = context.images_on_nodes[node.name][image_name]
                        bandwidth_usage += image_state.size[node.labels['beta.kubernetes.io/arch']]
                    except KeyError:
                        pass

        # Add the storage data usage
        bandwidth_usage += parse_size_string(pod.spec.labels.get('data.skippy.io/receives-from-storage', '0'))
        bandwidth_usage += parse_size_string(pod.spec.labels.get('data.skippy.io/sends-to-storage', '0'))

        return 'bandwidth_usage', str(bandwidth_usage)


class CostOracle(Oracle):
    execution_time_oracle = ExecutionTimeOracle()

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'cost', None
        cost = 0
        labels = scheduling_result.suggested_host.labels
        if 'locality.skippy.io/type' in labels and labels['locality.skippy.io/type'] == 'cloud':
            _, time_str = self.execution_time_oracle.estimate(context, pod, scheduling_result)
            # TODO implement a more sophisticated model for the pricing based on the list:
            #  - https://aws.amazon.com/lambda/pricing/
            cost = 0.000001667 * 10 * float(time_str)
        return 'cost', str(cost)


class ResourceUtilizationOracle(Oracle):
    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'resource_utilization', None
        # TODO maybe a more sophisticated solution is necessary?
        # Calculate the fraction of used edge resources
        resource_utilization = 0
        node = scheduling_result.suggested_host
        labels = scheduling_result.suggested_host.labels
        if 'locality.skippy.io/type' in labels and labels['locality.skippy.io/type'] == 'edge':
            resource_utilization = self.score_resource_utilization(pod, node)
        return 'resource_utilization', str(resource_utilization)

    def score_resource_utilization(self, pod, node) -> float:
        mem_cap = node.capacity.memory
        cpu_cap = node.capacity.cpu_millis
        mem_all = 0
        cpu_all = 0
        for container in pod.spec.containers:
            cpu_all += container.resources.requests.get('cpu', container.resources.default_milli_cpu_request)
            mem_all += container.resources.requests.get('memory', container.resources.default_mem_request)
        return (mem_cap / mem_all) + (cpu_cap / cpu_all)
