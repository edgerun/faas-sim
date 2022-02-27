import logging
import pickle
import time
from typing import List

from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import LBConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import OsmoticLoadBalancerCapableFaasSystemFactory
from ext.jjnp21.automator.factories.function_scheduler import RandomFunctionSchedulerFactory
from ext.jjnp21.automator.factories.lb_scaler import OsmoticLoadBalancerScalerFactory, FractionLoadBalancerScalerFactory
from ext.jjnp21.automator.factories.lb_scheduler import OsmoticLoadBalancerSchedulerFactory, \
    EverywhereLoadBalancerSchedulerFactory
from ext.jjnp21.automator.main import ExperimentRunAutomator
from ext.jjnp21.experiments.osmotic.util import RealTopoType, get_topology_factory_for_type, \
    topo_type_to_printable_string
from ext.jjnp21.topologies.accurate_urban import City
from sim.topology import Topology

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


def get_experiment_pair(rps: int, duration: int, topo_type: RealTopoType, lb_target_fraction: float,
                        osmotic_pressure_threshold: float, osmotic_hysteresis: float) -> List[Experiment]:
    osmotic = Experiment(f'{rps}rps_{duration}s_{topo_type_to_printable_string(topo_type)}_osmotic',
                         lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                         lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                         client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                         client_placement_strategy=ClientPlacementStrategy.NONE,
                         benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                         faas_system_factory=OsmoticLoadBalancerCapableFaasSystemFactory(),
                         net_mode=NetworkSimulationMode.ACCURATE,
                         function_scheduler_factory=RandomFunctionSchedulerFactory(),
                         lb_scaler_factory=OsmoticLoadBalancerScalerFactory(
                             pressure_threshold=osmotic_pressure_threshold, hysteresis=osmotic_hysteresis),
                         lb_scheduler_factory=OsmoticLoadBalancerSchedulerFactory(),
                         function_scaling_strategy=FunctionScalingStrategy.AVG_REQUEST_RATE,
                         topology_factory=get_topology_factory_for_type(topo_type, seed=45, client_ratio=0.6))

    lrt = Experiment(f'{rps}rps_{duration}s_{topo_type_to_printable_string(topo_type)}_lrt_baseline',
                     lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                     lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                     client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                     client_placement_strategy=ClientPlacementStrategy.NONE,
                     benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                     faas_system_factory=OsmoticLoadBalancerCapableFaasSystemFactory(),
                     net_mode=NetworkSimulationMode.ACCURATE,
                     function_scheduler_factory=RandomFunctionSchedulerFactory(),
                     lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=lb_target_fraction),
                     lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
                     function_scaling_strategy=FunctionScalingStrategy.AVG_REQUEST_RATE,
                     topology_factory=get_topology_factory_for_type(topo_type, seed=45, client_ratio=0.6))

    rr = Experiment(f'{rps}rps_{duration}s_{topo_type_to_printable_string(topo_type)}_rr_baseline',
                    lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                    lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                    client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                    client_placement_strategy=ClientPlacementStrategy.NONE,
                    benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'RR'),
                    faas_system_factory=OsmoticLoadBalancerCapableFaasSystemFactory(),
                    net_mode=NetworkSimulationMode.ACCURATE,
                    function_scheduler_factory=RandomFunctionSchedulerFactory(),
                    lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=lb_target_fraction),
                    lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
                    function_scaling_strategy=FunctionScalingStrategy.AVG_REQUEST_RATE,
                    topology_factory=get_topology_factory_for_type(topo_type, seed=45, client_ratio=0.6))

    return [osmotic, lrt, rr]


def get_parametrization_set(rps=25, duration=2000, topo_type=RealTopoType.CITY, lb_target_fraction=0.05,
                            osmotic_pressure_threshold=0.035, osmotic_hysteresis=0.02):
    return (rps, duration, topo_type, lb_target_fraction, osmotic_pressure_threshold, osmotic_hysteresis)


def get_experiments() -> List[Experiment]:
    exps = []
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=25, topo_type=RealTopoType.CITY)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=50, topo_type=RealTopoType.CITY)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=75, topo_type=RealTopoType.CITY)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=25, topo_type=RealTopoType.NATION)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=50, topo_type=RealTopoType.NATION)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=75, topo_type=RealTopoType.NATION)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=25, topo_type=RealTopoType.GLOBAL)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=50, topo_type=RealTopoType.GLOBAL)))
    exps.extend(get_experiment_pair(*get_parametrization_set(rps=75, topo_type=RealTopoType.GLOBAL)))
    return exps


experiment_list = get_experiments()[:2]

automator = ExperimentRunAutomator(experiment_list, worker_count=8)
print('Running nation benchmark')
start = time.time()
results = automator.run()
end = time.time()
print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')

analyzer = BasicResultAnalyzer(results)
analysis_df = analyzer.basic_kpis()
lb_placement_df = analyzer.lb_placement()
print(lb_placement_df.to_markdown())

md = analysis_df.to_markdown()
print(md)

print('dumping results')
f = open('/home/jp/Documents/tmp/test.dump', 'wb')
# f = open('/home/jp/Documents/tmp/osmotic_basic_openfaas_scaling.dump', 'wb')
pickle.dump(results, f)
print('successfully dumped results')
f.flush()
f.close()
print('successfully dumped results')
