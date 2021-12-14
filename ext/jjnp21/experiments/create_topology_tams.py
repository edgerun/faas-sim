from ext.jjnp21.automator.factories.topology import *
from ext.jjnp21.debugging.tiny_topology_factory import TinyUrbanSensingTopologyFactory
from ext.jjnp21.ether_customization.export import export_to_tam_json
from ext.jjnp21.experiments.net_vis import draw_custom
import matplotlib.pyplot as plt

global_factory = GlobalDistributedRealisticCityFactory(seed=45, client_ratio=0.6)
nation_factory = NationDistributedRealisticCityFactory(seed=45, client_ratio=0.6)
city_factory = SingleRealisticCityFactory(seed=45, client_ratio=0.6)

global_topo = global_factory.create()
nation_topo = nation_factory.create()
city_topo = city_factory.create()
tiny_topo = TinyUrbanSensingTopologyFactory(client_ratio=0.6).create()

export_path = '/home/jp/Documents/GitProjects/tam/data'

# export_to_tam_json(city_topo, f'{export_path}/city.json', lambda x: 0)
plt.figure(figsize=(9, 8), dpi=300)

draw_custom(tiny_topo)
plt.show()
print('done')