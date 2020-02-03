"""
Fast creation of various topologies.
"""

import pprint
from typing import List

from core.model import Node
from sim.net import Topology, Edge, Link


def create_lan(nodes: List[Node], downlink_bw, uplink_bw, internal_bw=1000, name=None):
    edges = list()
    n = len(nodes)

    uplink = Link(uplink_bw, tags={'type': 'uplink', 'name': name})
    downlink = Link(downlink_bw, tags={'type': 'downlink', 'name': name})

    # save node specific links so we can link them together point-to-point later
    node_links = list()

    # create a link for each node (this basically means that each node has one ethernet device)
    # connect each node's link to up/down
    for i in range(n):
        node = nodes[i]
        node_link = Link(internal_bw, tags={'type': 'node', 'name': node.name})
        edges.append(Edge(node, node_link, directed=False))

        # edges.append(Edge(node_link, uplink, directed=True))
        # edges.append(Edge(downlink, node_link, directed=True))

        node_links.append(node_link)

    # # connect all nodes together (essentially like a switch)
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         link_i = node_links[i]
    #         link_j = node_links[j]
    #         edges.append(Edge(link_i, link_j, directed=False))

    # connect each node's link to a 'switch' (a node with infinite bandwidth)
    switch = f'switch_{name or id(nodes)}'
    for node in node_links:
        edges.append(Edge(node, switch))
    # connect switch to up/downlink (like a router)
    edges.append(Edge(switch, uplink, directed=True))
    edges.append(Edge(downlink, switch, directed=True))

    return edges, uplink, downlink


def create_wifi(nodes: List[Node], downlink_bw, uplink_bw, internal_bw=100, name=None):
    edges = list()

    wifi = Link(bandwidth=internal_bw, tags={'type': 'wifi', 'name': name})
    uplink = Link(uplink_bw, tags={'type': 'uplink', 'name': name})
    downlink = Link(downlink_bw, tags={'type': 'downlink', 'name': name})

    # connect each node to the wifi link
    for node in nodes:
        edges.append(Edge(node, wifi, directed=False))

    # connect the wifi to up/downlink
    edges.append(Edge(wifi, uplink, directed=True))
    edges.append(Edge(downlink, wifi, directed=True))

    return edges, uplink, downlink


def _example():
    nodes1 = [Node('a1'), Node('b1'), Node('c1')]
    edges1, uplink1, downlink1 = create_lan(nodes1, 30, 10, name='lan1')

    nodes2 = [Node('d2'), Node('e2'), Node('f2')]
    edges2, uplink2, downlink2 = create_lan(nodes2, 50, 50, name='lan2')  # symmetric internet

    nodes3 = [Node('xw'), Node('yw'), Node('zw')]
    edges3, uplink3, downlink3 = create_wifi(nodes3, 30, 10, name='wifi3')

    # all nodes
    nodes = nodes1 + nodes2 + nodes3
    edges = edges1 + edges2 + edges3

    ups = [uplink1, uplink2, uplink3]
    downs = [downlink1, downlink2, downlink3]

    # connect uplinks and downlinks to the 'internet'. this is a 'fake' node that simplifies connecting everything
    # together. will be filtered out by Topology.get_route, thereby simulating infinite bandwidth.
    internet = 'internet'
    for up in ups:
        edges.append(Edge(up, internet))
    for down in downs:
        edges.append(Edge(internet, down))

    t = Topology(nodes, edges)

    print(len(edges))

    r = t.get_route(nodes1[0], nodes1[2])
    print([link.tags for link in r.hops])

    r = t.get_route(nodes1[0], nodes2[2])
    print([link.tags for link in r.hops])

    r = t.get_route(nodes1[0], nodes3[1])
    print([link.tags for link in r.hops])

    pprint.pprint(t.create_bandwidth_graph())


if __name__ == '__main__':
    _example()
