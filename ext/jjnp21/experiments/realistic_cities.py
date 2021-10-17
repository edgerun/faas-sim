import logging
import time
from typing import List

from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.execution import run_experiment
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import LBConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import LocalizedLoadBalancerFaaSFactory
from ext.jjnp21.automator.factories.function_scheduler import RandomFunctionSchedulerFactory
from ext.jjnp21.automator.factories.lb_scaler import FractionLoadBalancerScalerFactory, \
    ManualSetLoadBalancerScalerFactory
from ext.jjnp21.automator.factories.lb_scheduler import EverywhereLoadBalancerSchedulerFactory, \
    CentralLoadBalancerSchedulerFactory
from ext.jjnp21.automator.factories.topology import GlobalDistributedUrbanSensingFactory, SingleRealisticCityFactory, \
    NationDistributedRealisticCityFactory, GlobalDistributedRealisticCityFactory
from ext.jjnp21.automator.main import ExperimentRunAutomator
from ext.jjnp21.debugging.tiny_topology_factory import TinyUrbanSensingTopologyFactory
from ext.jjnp21.topologies.accurate_urban import City
from sim.topology import Topology

logging.basicConfig(level=logging.INFO)
rps = 25
duration = 1500
client_ratio = 1

class TopologyType(Enum):
    CITY = 1
    NATION = 2
    GLOBAL = 3

def get_experiment_set(topo_type: TopologyType, seed: int) -> List[Experiment]:
    def get_topo():
        if topo_type == TopologyType.CITY:
            return SingleRealisticCityFactory(client_ratio=client_ratio, seed=seed)
        elif topo_type == TopologyType.NATION:
            return NationDistributedRealisticCityFactory(client_ratio=client_ratio, seed=seed)
        elif topo_type == TopologyType.GLOBAL:
            return GlobalDistributedRealisticCityFactory(client_ratio=client_ratio, seed=seed)

    lrtc = Experiment('Least Response Time centralized',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.ACCURATE,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=ManualSetLoadBalancerScalerFactory(target_count=1),
                lb_scheduler_factory=CentralLoadBalancerSchedulerFactory(),
                topology_factory=get_topo())

    lrtd = Experiment('Least Response Time on all nodes',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.ACCURATE,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=0.05),
                lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
                topology_factory=get_topo())

    rrc = Experiment('Round Robin centralized',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'RR'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.ACCURATE,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=ManualSetLoadBalancerScalerFactory(target_count=1),
                lb_scheduler_factory=CentralLoadBalancerSchedulerFactory(),
                topology_factory=get_topo())

    rrd = Experiment('Round Robin on all nodes',
                seed=42,
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'RR'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.ACCURATE,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=1),
                lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
                topology_factory=get_topo())

    return [lrtc, lrtd, rrc, rrd]


experiment_list = get_experiment_set(TopologyType.NATION, 42)
automator = ExperimentRunAutomator(experiment_list, worker_count=4)
print('Running nation benchmark')
start = time.time()
results = automator.run()
end = time.time()
print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')
# results.sort('experiment.name')
for r in results:
    print(f'Ran "{r.experiment.name}" in {r.run_duration_seconds}s')

analyzer = BasicResultAnalyzer(results)
analysis_df = analyzer.basic_kpis()
md = analysis_df.to_markdown()
print(md)

# analysis_df.to_csv('/home/jp/Documents/tmp/analysis.csv', sep=';')
# print('successfully ran analysis')
# print('dumping results')
# f = open('/home/jp/Documents/tmp/results.dump', 'wb')
# pickle.dump(results, f)
# f.flush()
# f.close()
# print('successfully dumped results')
