from collections import defaultdict
from typing import Optional

import ether.topology
from ether.core import Node, Connection

DockerRegistry = Node('registry')

class Topology(ether.topology.Topology):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self._node_index = dict()

    def init_docker_registry(self):
        """
        Attaches the global "dockerhub.com" DockerRegistry to all internet helper nodes in the topology.
        """
        if DockerRegistry not in self.nodes:
            self.add_node(DockerRegistry)
        for node in self.nodes:
            if isinstance(node, str) and node.startswith('internet'):
                self.add_connection(Connection(node, DockerRegistry))

    def route_by_node_name(self, source_name: str, destination_name: str):
        """
        Resolves a route between compute nodes given their names. Raises a value error if either source or destination
        do not exist.

        :param source_name: the source node name
        :param destination_name: the destination node name
        :return: a Route
        """
        source = self.find_node(source_name)
        if source is None:
            raise ValueError('source node not found: ' + source_name)

        destination = self.find_node(destination_name)
        if destination is None:
            raise ValueError('destination node not found: ' + destination_name)

        return self.route(source, destination)

    def find_node(self, node_name: str) -> Optional[Node]:
        """
        Looks up a compute node by its unique name.

        :param node_name: the node name
        :return: the node or none if it does not exist
        """
        if node_name in self._node_index:
            return self._node_index[node_name]

        for node in self.get_nodes():
            if node.name == node_name:
                self._node_index[node_name] = node
                return node

        return None


class LazyBandwidthGraph:
    """
    Behaves like a two-dimensional dictionary that lazily resolves the bandwidth between nodes.
    Can be called like this:

    >>> bw = LazyBandwidthGraph(topology)
    >>> bw['server_0']['dockerhub.com'] == 1000 # will resolve the route
    >>> bw['server_0']['dockerhub.com'] == 1000 # served from the cache
    """
    topology: Topology

    def __init__(self, topology: Topology) -> None:
        super().__init__()
        self.cache = defaultdict(dict)
        self.topology = topology

    def __getitem__(self, source):
        return self._Resolver(self, source)

    class _Resolver:
        def __init__(self, bwg: 'LazyBandwidthGraph', source: str) -> None:
            super().__init__()
            self.bwg = bwg
            self.source = source

        def __getitem__(self, destination: str) -> Optional[float]:
            if destination in self.bwg.cache[self.source]:
                return self.bwg.cache[self.source][destination]

            if self.source == destination:
                # FIXME: should this case maybe be handled in the scheduler/priorities?
                return 1.25e+8

            route = self.bwg.topology.route_by_node_name(self.source, destination)
            if not route or not route.hops:
                return None

            bandwidth = min([link.bandwidth for link in route.hops])

            self.bwg.cache[self.source][destination] = bandwidth
            return bandwidth
