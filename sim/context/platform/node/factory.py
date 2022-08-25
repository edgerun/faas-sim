from typing import Optional

from ether.core import Node as EtherNode
from faas.context import InMemoryNodeService
from faas.system import FunctionNode, NodeState
from faas.util.constant import zone_label
from skippy.core.model import Node as SkippyNode

from sim.context.platform.node.model import SimFunctionNode
from sim.core import Environment
from sim.skippy import SimulationClusterContext
from sim.topology import Topology


def create_node_service(env: Environment, topology: Topology):
    zones = set()
    nodes = []

    for node in topology.get_nodes():
        node_zone = node.labels.get(zone_label)
        if node_zone is not None:
            zones.add(node_zone)
        sim_fn_node = _create_sim_node(node, env, topology)
        if sim_fn_node is not None:
            nodes.append(sim_fn_node)
    zones = list(zones)
    return InMemoryNodeService[SimFunctionNode](zones, nodes)


def _get_bandwidth(node_name: str, topology: Topology) -> int:
    link_name = f'link_{node_name}'
    for link in topology.get_links():
        if link.tags['name'] == link_name:
            return link.bandwidth


def _create_sim_node(node: EtherNode, env: Environment, topology: Topology) -> Optional[SimFunctionNode]:
    cluster: SimulationClusterContext = env.cluster
    skippy_node: SkippyNode = cluster.get_node(node.name)
    if skippy_node is None:
        return None
    labels = node.labels.copy()
    labels.update(skippy_node.labels)
    fn_node = FunctionNode(
        name=node.name,
        arch=node.arch,
        cpus=node.capacity.cpu_millis / 1000,
        ram=node.capacity.memory,
        netspeed=_get_bandwidth(node.name, topology),
        labels=labels,
        allocatable={'cpu': f'{node.capacity.cpu_millis}m', 'memory': node.capacity.memory},
        cluster=labels.get(zone_label),
        state=NodeState.READY
    )

    degradation_model = env.degradation_models.get(node.name, None)

    sim_fn_node = SimFunctionNode(fn_node=fn_node)
    sim_fn_node.env = env
    sim_fn_node.ether_node = node
    sim_fn_node.skippy_node = skippy_node
    sim_fn_node.performance_degradation = degradation_model

    return sim_fn_node
