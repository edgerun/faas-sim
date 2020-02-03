import glob
import os
from ast import literal_eval as make_tuple
from typing import Tuple, NamedTuple

import pandas as pd

from core.clustercontext import ClusterContext
from core.model import Pod, SchedulingResult, ImageState
from core.utils import parse_size_string, normalize_image_name
from sim.oracle.data.distributions import execution_time_distributions, startup_time_distributions
from sim.stats import BoundRejectionSampler, BufferedSampler

Bandwidth = NamedTuple('Bandwidth', [('mbit', int), ('delay', int), ('deviation', int)])

data_dir = 'sim/oracle/data'


class Oracle:
    """Abstract class for startup oracle functions."""

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


class StartupTimeOracle(EmpiricalOracle):
    def __init__(self):
        super(StartupTimeOracle, self).__init__(os.path.join(data_dir, 'pod_startup_*.csv'))
        self.durations = self.dataset[['host', 'bandwidth', 'image', 'image_present', 'duration']]

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'startup_time', None
        host = scheduling_result.suggested_host.name
        host_type = host[host.rindex('_') + 1:]
        # For the startup time the bandwidth to the registry is necessary
        bandwidth = context.get_bandwidth_graph()[host]['registry']
        startup_time = 0

        for container in pod.spec.containers:
            image = container.image
            image_present = normalize_image_name(image) not in scheduling_result.needed_images

            data = self.durations.query(f'host == "{host_type}" and '
                                        f'image == "{image}" and '
                                        f'bandwidth == "{bandwidth}" and '
                                        f'image_present == {image_present}')

            if data.empty:
                raise ValueError('no data for %s, %s, %s, %s' % (host_type, image, bandwidth, image_present))
            else:
                sample = data['duration'].sample()

            startup_time += sample.values[0]

        return 'startup_time', str(startup_time)


class ExecutionTimeOracle(EmpiricalOracle):
    def __init__(self):
        super(ExecutionTimeOracle, self).__init__(os.path.join(data_dir, 'exec_time*.csv'))
        self.durations = self.dataset[['host', 'bandwidth', 'image', 'duration']]

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'execution_time', None
        host = scheduling_result.suggested_host.name
        host_type = host[host.rindex('_') + 1:]
        # For the execution time the bandwidth to the next storage node is necessary
        bandwidth = context.get_bandwidth_graph()[host][context.get_next_storage_node(scheduling_result.suggested_host)]
        execution_time = 0
        for container in pod.spec.containers:
            image = container.image
            execution_time += self.durations.query(f'host == "{host_type}" and '
                                                   f'bandwidth == {bandwidth} and '
                                                   f'image == "{image}"')['duration'].sample().values[0]
        return 'execution_time', str(execution_time)


class BandwidthUsageOracle(Oracle):
    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'bandwidth_usage', None

        # Calculate the image pull bandwidth
        bandwidth_usage = 0
        node = scheduling_result.suggested_host
        for image_name in scheduling_result.needed_images:
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
    execution_time_oracle: Oracle

    def __init__(self, execution_time_oracle=None) -> None:
        super().__init__()
        self.execution_time_oracle = execution_time_oracle or FittedExecutionTimeOracle()

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'cost', None
        cost = 0
        labels = scheduling_result.suggested_host.labels
        if 'locality.skippy.io/type' in labels and labels['locality.skippy.io/type'] == 'cloud':
            _, time_str = self.execution_time_oracle.estimate(context, pod, scheduling_result)
            # TODO implement a more sophisticated model for the pricing based on the list:
            #  - https://aws.amazon.com/lambda/pricing/
            # 0.000001667 is the cost for a function exec with 1GB RAM per 100ms
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
        return (mem_all / mem_cap) + (cpu_all / cpu_cap)


class FittedStartupTimeOracle(Oracle):

    def __init__(self) -> None:
        super().__init__()
        self.startup_time_samplers = {
            k: BoundRejectionSampler(BufferedSampler(dist), xmin, xmax) for k, (xmin, xmax, dist) in
            startup_time_distributions.items()
        }

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'startup_time', None

        host = scheduling_result.suggested_host.name
        host_type = host[host.rindex('_') + 1:]
        # For the startup time the bandwidth to the registry is necessary
        bandwidth = context.get_bandwidth_graph()[host]['registry']
        startup_time = 0

        for container in pod.spec.containers:
            image = container.image

            image_present = normalize_image_name(image) not in scheduling_result.needed_images

            k = (host_type, image, image_present, bandwidth)

            if k not in self.startup_time_samplers:
                raise ValueError(k)

            startup_time += self.startup_time_samplers[k].sample()

        return 'startup_time', str(startup_time)


class HackedFittedStartupTimeOracle(Oracle):
    """
    Always uses 1 Gbit preset, and subtracts the theoretical transmission time, to give an estimate of the actual
    container startup time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.startup_time_samplers = {
            k: BoundRejectionSampler(BufferedSampler(dist), xmin, xmax) for k, (xmin, xmax, dist) in
            startup_time_distributions.items()
        }

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'startup_time', None

        host = scheduling_result.suggested_host.name
        host_arch = scheduling_result.suggested_host.labels['beta.kubernetes.io/arch']
        host_type = host[host.rindex('_') + 1:]
        bandwidth = int(1.25e7)  # always assume 100mbit (which is probably the downlink we have @ DSG)
        startup_time = 0

        for container in pod.spec.containers:
            image = container.image
            image_name = normalize_image_name(image)

            image_present = image_name not in scheduling_result.needed_images

            k = (host_type, image, image_present, bandwidth)

            if k not in self.startup_time_samplers:
                raise ValueError(k)

            image_time = self.startup_time_samplers[k].sample()

            if not image_present:
                image_size = context.get_image_state(image_name).size[host_arch]
                dl_time = image_size / bandwidth
                image_time = max(0, image_time - dl_time)

            startup_time += image_time

        return 'startup_time', str(startup_time)


class FittedExecutionTimeOracle(Oracle):

    def __init__(self) -> None:
        super().__init__()
        self.execution_time_samplers = {
            k: BoundRejectionSampler(BufferedSampler(dist), xmin, xmax) for k, (xmin, xmax, dist) in
            execution_time_distributions.items()
        }

    def estimate(self, context: ClusterContext, pod: Pod, scheduling_result: SchedulingResult) -> Tuple[str, str]:
        if scheduling_result is None or scheduling_result.suggested_host is None:
            return 'execution_time', None

        host = scheduling_result.suggested_host.name
        host_type = host[host.rindex('_') + 1:]
        # For the execution time the bandwidth to the next storage node is necessary
        bandwidth = context.get_bandwidth_graph()[host][context.get_next_storage_node(scheduling_result.suggested_host)]

        execution_time = 0
        for container in pod.spec.containers:
            image = container.image

            k = (host_type, image, bandwidth)
            if k not in self.execution_time_samplers:
                raise ValueError(k)

            # currently this works because we assume only one container (the function) per pod
            execution_time += self.execution_time_samplers[k].sample()

        return 'execution_time', str(execution_time)
