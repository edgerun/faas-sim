from collections import defaultdict
from typing import List

from skippy.core.clustercontext import BandwidthGraph
from skippy.core.model import Node

# 1.25e+6 Byte/s = 10 MBit/s
# 1.25e+7 Byte/s = 100 MBit/s
# 1.25e+8 Byte/s = 1 GBit/s
# 1.25e+9 Byte/s = 10 GBit/s - assumed for local access
bandwidth_types = {
    'edge': {
        'edge': 1.25e+6,  # Edge-to-Edge: 10 MBit/s
        'cloud': 1.25e+7  # Edge-to-Cloud: 100 MBit/s
    },
    'cloud': {
        'edge': 1.25e+7,  # Cloud-to-Edge: 100 MBit/s
        'cloud': 1.25e+8  # Cloud-to-Cloud: 1 GBit/s
    }
}


def generate_bandwidth_graph(nodes: List[Node]) -> BandwidthGraph:
    bandwidth_graph: BandwidthGraph = defaultdict(lambda: defaultdict())

    # Create the complete graph for the nodes with a bandwidth depending on the type
    for node in nodes:
        for inner in nodes:
            bandwidth_graph[node.name][inner.name] = 1.25e+8 if node == inner else \
                bandwidth_types[node.labels['locality.skippy.io/type']][inner.labels['locality.skippy.io/type']]

    # Also add a connection from each node to the registry
    for node in nodes:
        # Bandwidth to the registry is 100 MBit/s for cloud, 10 MBit/s for edge
        bandwidth = 1.25e+7 if node.labels['locality.skippy.io/type'] == 'cloud' else 1.25e+6
        bandwidth_graph[node.name]['registry'] = bandwidth

    return bandwidth_graph
