import logging
import time

from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.execution import run_experiment
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import LBConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import OsmoticLoadBalancerCapableFaasSystemFactory
from ext.jjnp21.automator.factories.function_scheduler import RandomFunctionSchedulerFactory
from ext.jjnp21.automator.factories.lb_scaler import OsmoticLoadBalancerScalerFactory, FractionLoadBalancerScalerFactory
from ext.jjnp21.automator.factories.lb_scheduler import OsmoticLoadBalancerSchedulerFactory, \
    EverywhereLoadBalancerSchedulerFactory
from ext.jjnp21.automator.factories.topology import GlobalDistributedUrbanSensingFactory, \
    GlobalDistributedRealisticCityFactory
from ext.jjnp21.debugging.tiny_topology_factory import TinyUrbanSensingTopologyFactory
from ext.jjnp21.experiments.net_vis import draw_custom
from ext.jjnp21.topologies.accurate_urban import City
from sim.topology import Topology
import matplotlib.pyplot as plt
import pickle

logging.basicConfig(level=logging.INFO)
rps = 25
duration = 2000


class AccurateCityTopologyFactory(TopologyFactory):
    def create(self) -> Topology:
        topology = Topology()
        city = City(cloud_node_count=1,
                    client_count=100,
                    smart_pole_count=15,
                    cell_tower_count=20,
                    five_g_share=0.1,
                    cell_tower_compute_share=1,
                    internet='internet',
                    seed=42)
        city.materialize(topology)
        topology.init_docker_registry()
        return topology

exp_old = Experiment('Least Response Time on all nodes',
                 lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                 lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                 client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                 client_placement_strategy=ClientPlacementStrategy.NONE,
                 benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                 faas_system_factory=OsmoticLoadBalancerCapableFaasSystemFactory(),
                 net_mode=NetworkSimulationMode.ACCURATE,
                 function_scheduler_factory=RandomFunctionSchedulerFactory(),
                 lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=0.2),
                 lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
                 topology_factory=GlobalDistributedRealisticCityFactory(seed=45, client_ratio=0.6))

exp = Experiment('Least Response Time on all nodes',
                 lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                 lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                 client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                 client_placement_strategy=ClientPlacementStrategy.NONE,
                 benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                 faas_system_factory=OsmoticLoadBalancerCapableFaasSystemFactory(),
                 net_mode=NetworkSimulationMode.ACCURATE,
                 function_scheduler_factory=RandomFunctionSchedulerFactory(),
                 lb_scaler_factory=OsmoticLoadBalancerScalerFactory(pressure_threshold=0.02, hysteresis=0.05),
                 lb_scheduler_factory=OsmoticLoadBalancerSchedulerFactory(),
                 topology_factory=GlobalDistributedRealisticCityFactory(seed=45, client_ratio=0.6))
                 # topology_factory=GlobalDistributedUrbanSensingFactory(client_ratio=0.6))
# topo = TinyUrbanSensingTopologyFactory(client_ratio=0.6).create()
# draw_custom(topo)
# plt.show()

start = time.time()
result = run_experiment(exp)
end = time.time()
print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')

analyzer = BasicResultAnalyzer([result])
analysis_df = analyzer.basic_kpis()
md = analysis_df.to_markdown()
print(md)

#
# analysis_df.to_csv('/home/jp/Documents/tmp/analysis.csv', sep=';')
# print('successfully ran analysis')
print('dumping results')
f = open('/home/jp/Documents/tmp/debug_results.dump', 'wb')
pickle.dump(result, f)
print('successfully dumped results')
# f.flush()
# f.close()
# print('successfully dumped results')
