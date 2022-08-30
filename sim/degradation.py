import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
from faas.system import FunctionReplica

from .oracle.oracle import ResourceOracle


@dataclass
class DegradationTrace:
    replica: FunctionReplica
    start: float
    end: float

def create_degradation_model_input(calls: List[DegradationTrace], start_ts, end_ts, node_name: str,
                                   ram_capacity: float,
                                   resource_oracle: ResourceOracle) -> np.ndarray:
    # input of model is an array with 34 elements
    # in general, the input is based on the resource usages that occurred during the function execution
    # for each trace (instance) from the target service following metrics
    # for each resource, the millis of usage per image get recorded
    # i.e. if image A has 2 requests, during the execution and each call had 200 millis CPU, the input for this
    # image is the sum of the millis = 400.
    # after having summed up all usages per image, the input is formed by calculating the following measures:
    # mean, std dev, min, max, 25 percentile, 50 percentile and 75 percentile
    # this translates to the following indices
    # ['cpu', 'gpu', 'blkio', 'net']
    # 0 - 6: cpu mean, cpu std dev,...
    # 7 - 13: gpu mean, gpu std dev...
    # 14 - 20: blkio mean, blkio std dev...
    # 21 - 27: net mean, net std dev...
    # 28: number of running containers that have executed at least one call
    # 29: sum of all cpu millis
    # 30: sum of all gpu millis
    # 31: sum of all blkio rate ! not scaled
    # 32: sum of all net rate ! not scaled
    # 33: mean ram percentage over complete experiment
    resources_types = ['cpu', 'gpu', 'blkio', 'net']

    if len(calls) == 0:
        return np.array([])
    ram = 0
    seen_replicas = set()
    resources = defaultdict(lambda: defaultdict(list))
    for call in calls:
        function = call.replica.function.name
        replica_id = call.replica.replica_id
        call_resources = resource_oracle.get_resources(node_name, function)
        if call_resources:
            raise ValueError(f"Can't find resources for node '{node_name}' for function {function}")
        for resource_type in resources_types:
            resources[replica_id][resource_type].append(call_resources[resource_type])
        if len(call_resources) == 0:
            logging.debug(f'Function {function.name} has no resources for node {node_name}')
            continue

        if replica_id not in seen_replicas:
            ram += call.replica.container.get_resource_requirements()['memory']
            seen_replicas.add(replica_id)
        last_start = start_ts if start_ts >= call.start else call.start

        if call.end is not None:
            first_end = end_ts if end_ts <= call.end else call.end
        else:
            first_end = end_ts

        overlap = first_end - last_start

        for resource in resources_types:
            resources[replica_id][resource].append(overlap * call_resources[resource])

    sums = defaultdict(list)
    for resource_type in resources_types:
        for replica_id, resources_of_pod in resources.items():
            resource_sum = np.sum(resources_of_pod[resource_type])
            sums[resource_type].append(resource_sum)

    # make input for model
    # the values get converted to a fixed length array, i.e.: descriptive statistics
    # of the resources of all faas containers
    # skip the first element, it's only the count of containers
    input = []
    for resource in resources_types:
        mean = np.mean(sums[resource])
        std = np.std(sums[resource])
        amin = np.min(sums[resource])
        amax = np.max(sums[resource])
        p_25 = np.percentile(sums[resource], q=0.25)
        p_50 = np.percentile(sums[resource], q=0.5)
        p_75 = np.percentile(sums[resource], q=0.75)
        for value in [mean, std, amin, amax, p_25, p_50, p_75]:
            # in case of only one container the std will be np.nan
            if np.isnan(value):
                input.append(0)
            else:
                input.append(value)

    # add number of containers
    input.append(len(sums['cpu']))

    # add total sums resources
    for resource in resources_types:
        input.append(np.sum(sums[resource]))

    # add ram_rate in percentage too
    input.append(ram / ram_capacity)

    return np.array(input)
