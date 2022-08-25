from faas.context import NetworkService

from sim.topology import Topology


class SimNetworkService(NetworkService):

    def __init__(self, topology: Topology):
        self.topology = topology

    def get_latency(self, from_node: str, to_node: str) -> float:
        from_ether_node = self.topology.find_node(from_node)
        to_ether_node = self.topology.find_node(to_node)
        return self.topology.latency(from_ether_node, to_ether_node)

    def get_max_latency(self) -> float:
        raise NotImplementedError()

    def update_latency(self, from_node: str, to_noe: str, value: float):
        raise NotImplementedError()
