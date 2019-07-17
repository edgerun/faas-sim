from collections import defaultdict
from typing import List

from core.clustercontext import BandwidthGraph
from core.model import Node


def generate_bandwidth_graph(nodes: List[Node]) -> BandwidthGraph:
    node_names = [node.name for node in nodes]
    bandwidth_graph: BandwidthGraph = defaultdict(lambda: defaultdict())
    for node in node_names:
        for inner in node_names:
            bandwidth_graph[node][inner] = 1.25e+9

    for node in node_names:
        bandwidth_graph[node]['registry'] = 1.25e+9
    '''
    # TODO synthesize the graph -> N^2 entries!
    #  for all nodes in self.list_nodes()
    # 1.25e+6 Byte/s = 10 MBit/s
    # 1.25e+7 Byte/s = 100 MBit/s
    # 1.25e9 Byte/s = 10 GBit/s - assumed for local access
    # The registry is always connected with 100 MBit/s (replicated in both networks)
    # The edge nodes are interconnected with 100 MBit/s
    # The cloud is connected to the edge nodes with 10 MBit/s
    return {
        'ara-clustercloud1': {
            'ara-clustercloud1': 1.25e+9,
            'ara-clustertegra1': 1.25e+6,
            'ara-clusterpi1': 1.25e+6,
            'ara-clusterpi2': 1.25e+6,
            'ara-clusterpi3': 1.25e+6,
            'ara-clusterpi4': 1.25e+6,
            'registry': 1.25e+7
        },
        'ara-clustertegra1': {
            'ara-clustercloud1': 1.25e+6,
            'ara-clustertegra1': 1.25e+9,
            'ara-clusterpi1': 1.25e+7,
            'ara-clusterpi2': 1.25e+7,
            'ara-clusterpi3': 1.25e+7,
            'ara-clusterpi4': 1.25e+7,
            'registry': 1.25e+7
        },
        'ara-clusterpi1': {
            'ara-clustercloud1': 1.25e+6,
            'ara-clustertegra1': 1.25e+7,
            'ara-clusterpi1': 1.25e+9,
            'ara-clusterpi2': 1.25e+7,
            'ara-clusterpi3': 1.25e+7,
            'ara-clusterpi4': 1.25e+7,
            'registry': 1.25e+7
        },
        'ara-clusterpi2': {
            'ara-clustercloud1': 1.25e+6,
            'ara-clustertegra1': 1.25e+7,
            'ara-clusterpi1': 1.25e+7,
            'ara-clusterpi2': 1.25e+9,
            'ara-clusterpi3': 1.25e+7,
            'ara-clusterpi4': 1.25e+7,
            'registry': 1.25e+7
        },
        'ara-clusterpi3': {
            'ara-clustercloud1': 1.25e+6,
            'ara-clustertegra1': 1.25e+7,
            'ara-clusterpi1': 1.25e+7,
            'ara-clusterpi2': 1.25e+7,
            'ara-clusterpi3': 1.25e+9,
            'ara-clusterpi4': 1.25e+7,
            'registry': 1.25e+7
        },
        'ara-clusterpi4': {
            'ara-clustercloud1': 1.25e+6,
            'ara-clustertegra1': 1.25e+7,
            'ara-clusterpi1': 1.25e+7,
            'ara-clusterpi2': 1.25e+7,
            'ara-clusterpi3': 1.25e+7,
            'ara-clusterpi4': 1.25e+9,
            'registry': 1.25e+7
        }
    }
    '''
    return bandwidth_graph
