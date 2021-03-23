from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from sim.faas import FunctionReplica


@dataclass
class ResourceWindow:
    replica: FunctionReplica
    cpu: float


class MetricsServer:
    """
    contains methods to obtain metrics - offers query functions for resources (functionreplica)

    stores time-series data in data structure (i.e. list)

    """

    def __init__(self):
        # TODO this will inevitably leak memory
        self._windows = defaultdict(lambda: defaultdict(list))

    def put(self, window: ResourceWindow):
        node = window.replica.node.name
        pod = window.replica.pod.name

        self._windows[node][pod].append(window)

    def get_average_cpu_utilization(self, fn_replica: FunctionReplica, window_size: int = 10) -> float:
        # use this for HPA
        node = fn_replica.node.name
        pod = fn_replica.pod.name
        windows = self._windows.get(node, {}).get(pod, [])
        if len(windows) == 0:
            return 0

        # slicing never throws IndexError
        return np.mean(list(map(lambda l: l.cpu, windows[-window_size:])))
