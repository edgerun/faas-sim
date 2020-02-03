import matplotlib.pyplot as plt
import networkx as nx

from core.model import Node
from sim.net import Topology, Link
from sim.scenarios import UrbanSensingClusterSynthesizer

synth = UrbanSensingClusterSynthesizer(cells=2, cloud_vms=1)

t: Topology = synth.create_topology()

G = nx.Graph()


def node_name(obj):
    if isinstance(obj, Node):
        return obj.name
    elif isinstance(obj, Link):
        return f'link_{id(obj)}'
    else:
        return str(obj)


for edge in t.edges:
    a = edge.source
    b = edge.target

    la = node_name(a)
    lb = node_name(b)

    G.add_edge(la, lb)

print(G.nodes)

pos = nx.spring_layout(G)  # positions for all nodes

# nodes

netnodes = [node for node in G.nodes if not str(node).startswith('link_') and not str(node).startswith('switch_')]
links = [node for node in G.nodes if str(node).startswith('link_')]
switches = [node for node in G.nodes if str(node).startswith('switch_')]
try:
    netnodes.remove('internet')
except ValueError:
    pass

nx.draw_networkx_nodes(G, pos,
                       nodelist=netnodes,
                       node_color='b',
                       node_size=300,
                       alpha=0.8)
nx.draw_networkx_nodes(G, pos,
                       nodelist=links,
                       node_color='r',
                       node_size=50,
                       alpha=0.8)
nx.draw_networkx_nodes(G, pos,
                       nodelist=switches,
                       node_color='y',
                       node_size=200,
                       alpha=0.8)
nx.draw_networkx_nodes(G, pos,
                       nodelist=['internet'],
                       node_color='g',
                       node_size=800,
                       alpha=0.8)

nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, dict(zip(netnodes, netnodes)), font_size=8)
plt.axis('off')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

plt.show()  # display
