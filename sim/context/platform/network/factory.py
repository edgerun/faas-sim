from faas.context import NetworkService

from sim.context.platform.network.service import SimNetworkService
from sim.topology import Topology


def create_network_service(topology: Topology) -> NetworkService:
    return SimNetworkService(topology)
