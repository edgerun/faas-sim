from typing import Callable

from ether.topology import Topology
from ether.core import Node
import json


def export_to_tam_json(topology: Topology, output_file: str, value_projector: Callable[[Node], int]):
    nodes = []
    links = []
    if value_projector is None:
        value_projector = lambda: 0
    for node in topology.nodes:
        if isinstance(node, str):
            nodes.append({
                'id': id(node),
                'name': node,
                'value': value_projector(node)
            })
            continue
        nodes.append({
            'id': id(node),
            'name': node.name if isinstance(node, Node) else node.tags['name'],
            'value': value_projector(node)
        })
    for edge in topology.edges.values():
        links.append({
            'source': id(edge['connection'].source),
            'target': id(edge['connection'].target),
            'directed': edge['directed']
        })
    full = {
        'nodes': nodes,
        'links': links
    }
    with open(output_file, 'w') as file:
        json.dump(full, file)
        file.flush()
        file.close()
