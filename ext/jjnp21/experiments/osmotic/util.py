from enum import Enum

from ext.jjnp21.automator.factories.topology import TopologyFactory, GlobalDistributedRealisticCityFactory, \
    NationDistributedRealisticCityFactory, SingleRealisticCityFactory


class RealTopoType(Enum):
    CITY = 'city'
    NATION = 'nation'
    GLOBAL = 'global'

def topo_type_to_printable_string(type: RealTopoType):
    if type == RealTopoType.CITY:
        return 'city'
    if type == RealTopoType.NATION:
        return 'nation'
    if type == RealTopoType.GLOBAL:
        return 'global'

def get_topology_factory_for_type(type: RealTopoType, client_ratio=0.6, seed=42) -> TopologyFactory:
    if type == RealTopoType.CITY:
        return SingleRealisticCityFactory(client_ratio=client_ratio, seed=seed)
    if type == RealTopoType.NATION:
        return NationDistributedRealisticCityFactory(client_ratio=client_ratio, seed=seed)
    if type == RealTopoType.GLOBAL:
        return GlobalDistributedRealisticCityFactory(client_ratio=client_ratio, seed=seed)

# def get_topo_type_friendly_name(type: RealTopoType) -> str:
#     if type == RealTopoType.CITY:
#         return 'city'
#     if type == RealTopoType.NATION:
#         return 'nation'
#     if type == RealTopoType.GLOBAL:
#         return 'global'