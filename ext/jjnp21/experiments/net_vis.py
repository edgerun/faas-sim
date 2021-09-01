import networkx as nx

from ether.core import Node, Link

from ext.jjnp21.automator.factories.topology import RaithHeterogeneousUrbanSensingFactory, \
    NationDistributedUrbanSensingFactory, GlobalDistributedUrbanSensingFactory
import matplotlib.pyplot as plt

from ext.jjnp21.topology import client_label


def draw_custom(topology):
    # pos = nx.spring_layout(topology, seed=42, iterations=150, k=0.05)  # positions for all nodes
    pos = nx.kamada_kawai_layout(topology)
    # nodes

    hosts = [node for node in topology.nodes if isinstance(node, Node) and node.labels.get(client_label, None) is None]
    clients = [node for node in topology.nodes if
               isinstance(node, Node) and node.labels.get(client_label, None) is not None]
    internet = [node for node in topology.nodes if isinstance(node, str) and node.startswith('internet')]
    links = [node for node in topology.nodes if isinstance(node, Link)]
    switches = [node for node in topology.nodes if str(node).startswith('switch_')]
    lbs = [node for node in topology.nodes if isinstance(node, Node) and node.name.startswith('load-balancer')]

    nx.draw_networkx_nodes(topology, pos,
                           nodelist=hosts,
                           node_color='b',
                           node_size=300,
                           alpha=0.8)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=links,
                           node_color='g',
                           node_size=50,
                           alpha=0.9)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=switches,
                           node_color='y',
                           node_size=200,
                           alpha=0.8)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=internet,
                           node_color='r',
                           node_size=800,
                           alpha=0.8)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=lbs,
                           node_color='orange',
                           node_size=800,
                           alpha=0.8)
    nx.draw_networkx_nodes(topology, pos,
                           nodelist=clients,
                           node_color='pink',
                           node_size=200,
                           alpha=0.8)

    nx.draw_networkx_edges(topology, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(topology, pos, dict(zip(hosts, hosts)), font_size=10)
    nx.draw_networkx_labels(topology, pos, dict(zip(internet, internet)), font_size=10)
    nx.draw_networkx_labels(topology, pos, dict(zip(links, [l.tags['type'] for l in links])), font_size=8)
    # nx.draw_networkx_labels(topology, pos, dict(zip(links, links)), font_size=8)


if __name__ == '__main__':
    topology = NationDistributedUrbanSensingFactory(client_ratio=0.6, seed=42).create()
    # topology = RaithHeterogeneousUrbanSensingFactory(node_count=50, client_ratio=0.3, seed=42).create()
    plt.figure(figsize=(60, 48), dpi=80)
    draw_custom(topology)
    plt.show()
