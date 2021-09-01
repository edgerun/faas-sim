from pathlib import Path
import json

from ext.jjnp21.automator.factories.topology import GlobalDistributedUrbanSensingFactory, \
    RaithHeterogeneousUrbanSensingFactory
from ext.jjnp21.debugging.tiny_topology_factory import TinyUrbanSensingTopologyFactory
from ext.jjnp21.experiments.net_vis import draw_custom
from matplotlib import pyplot as plt
import networkx as nx

# topology = TinyUrbanSensingTopologyFactory(client_ratio=1, node_count=20).create()
# topology = GlobalDistributedUrbanSensingFactory(client_ratio=0.6).create()
topology = RaithHeterogeneousUrbanSensingFactory(node_count = 150, client_ratio=0.6).create()

plt.figure(figsize=(30, 24), dpi=80)
# draw_custom(topology)
# plt.show()
cyto = nx.cytoscape_data(topology)


out_path = '/home/jp/Documents/tmp/vis'
Path(out_path).mkdir(parents=True, exist_ok=True)
# p = Path(f'{out_path}/city.gml').mkdir(parents=True, exist_ok=True)

# nx.write_gml(topology, f'{out_path}/city.gml', lambda x: str(x))

with open(f'{out_path}/city.json', 'w') as outfile:
    json.dump(cyto, outfile)
    print('wrote to json')
