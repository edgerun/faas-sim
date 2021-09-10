from pathlib import Path
import json

from ext.jjnp21.automator.factories.topology import GlobalDistributedUrbanSensingFactory, \
    RaithHeterogeneousUrbanSensingFactory
from ext.jjnp21.debugging.tiny_topology_factory import TinyUrbanSensingTopologyFactory
from ext.jjnp21.ether_customization.export import export_to_tam_json
from ext.jjnp21.experiments.net_vis import draw_custom
from matplotlib import pyplot as plt
import networkx as nx

# topology = TinyUrbanSensingTopologyFactory(client_ratio=1, node_count=20).create()
# topology = GlobalDistributedUrbanSensingFactory(client_ratio=0.6).create()
from ext.jjnp21.topologies.accurate_urban import create_city, City

# topology = RaithHeterogeneousUrbanSensingFactory(node_count = 50, client_ratio=0.6).create()
from sim.topology import Topology

plt.figure(figsize=(30, 24), dpi=80)
# draw_custom(topology)
# demo_city = create_city()
# draw_custom(demo_city)
# plt.show()
# export_to_tam_json(demo_city, '/home/jp/Desktop/demo.json', lambda n: 42)

city = City(cloud_node_count=20,
            client_count=100,
            smart_pole_count=15,
            cell_tower_count=20,
            five_g_share=0.2,
            cell_tower_compute_share=0.25,
            internet='internet',
            seed=42)
topo = Topology()
city.materialize(topo)
draw_custom(topo)
plt.show()
export_to_tam_json(topo, '/home/jp/Desktop/demo.json', lambda n: 42)


print("something")
# cyto = nx.cytoscape_data(topology)

#
# out_path = '/home/jp/Documents/tmp/vis'
# Path(out_path).mkdir(parents=True, exist_ok=True)
# p = Path(f'{out_path}/city.gml').mkdir(parents=True, exist_ok=True)

# nx.write_gml(topology, f'{out_path}/city.gml', lambda x: str(x))

# with open(f'{out_path}/city.json', 'w') as outfile:
#     json.dump(cyto, outfile)
#     print('wrote to json')
