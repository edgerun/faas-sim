import logging
import pickle
import time
from functools import partial
from typing import List, Dict, Any

from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import LBConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import OsmoticLoadBalancerCapableFaasSystemFactory, \
    PerRegionRequestOriginFaasSystemFactory
from ext.jjnp21.automator.factories.function_scheduler import RandomFunctionSchedulerFactory
from ext.jjnp21.automator.factories.lb_scaler import OsmoticLoadBalancerScalerFactory, FractionLoadBalancerScalerFactory
from ext.jjnp21.automator.factories.lb_scheduler import OsmoticLoadBalancerSchedulerFactory, \
    EverywhereLoadBalancerSchedulerFactory
from ext.jjnp21.automator.main import ExperimentRunAutomator
from ext.jjnp21.experiments.osmotic.util import RealTopoType, get_topology_factory_for_type, \
    topo_type_to_printable_string
from ext.jjnp21.topologies.accurate_urban import City
from sim.topology import Topology
import numpy as np

logging.basicConfig(level=logging.INFO)
rps = 25
duration = 2000

def mapping_fn(shift, now):
    wavelength = duration / 2.4
    if now < shift or now > shift + wavelength:
        return 0.005
    else:
        return np.sin((now - shift) * np.pi / wavelength)

def build_region_weight_mapping(region_list: List[str]) -> Dict[str, Any]:
    mappings = {}
    for index, region in enumerate(region_list):
        mappings[region] = partial(mapping_fn, index * duration / 2.8)
    return mappings

def create_experiment(threshold: float, hysteresis: float):
    return Experiment(f'{rps}rps_{duration}s_global_osmotic pthres={threshold} hyst={hysteresis}',
                      lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                      lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                      client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                      client_placement_strategy=ClientPlacementStrategy.NONE,
                      benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                      faas_system_factory=PerRegionRequestOriginFaasSystemFactory(region_weight_mapping=build_region_weight_mapping(['newyork', 'london', 'sydney'])),
                      net_mode=NetworkSimulationMode.ACCURATE,
                      function_scheduler_factory=RandomFunctionSchedulerFactory(),
                      lb_scaler_factory=OsmoticLoadBalancerScalerFactory(
                             pressure_threshold=threshold, hysteresis=hysteresis),
                      lb_scheduler_factory=OsmoticLoadBalancerSchedulerFactory(),
                      function_scaling_strategy=FunctionScalingStrategy.AVG_REQUEST_RATE,
                      topology_factory=get_topology_factory_for_type(RealTopoType.GLOBAL, seed=45, client_ratio=0.6))

def get_experiments() -> List[Experiment]:
    # (0.1, 0.05), (0.08, 0.04), (0.06, 0.03), (0.04, 0.02), (0.03, 0.015),
    vals = [(0.02, 0.2), (0.05, 0.08), (0.03, 0.05), (0.1, 0.15)]
    exps = []
    for (threshold, hysteresis) in vals:
        exps.append(create_experiment(threshold, hysteresis))
    return exps


experiment_list = get_experiments()

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
f = open('/home/jp/Documents/tmp/region_shift2.dump', 'wb')
# f = open('/home/jp/Documents/tmp/osmotic_basic_openfaas_scaling.dump', 'wb')
for r in results:
    r.experiment.faas_system_factory = None
pickle.dump(results, f)
print('successfully dumped results')
f.flush()
f.close()
print('successfully dumped results')
