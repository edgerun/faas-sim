import logging
import pickle
import time

from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import LBConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import LocalizedLoadBalancerFaaSFactory
from ext.jjnp21.automator.factories.function_scheduler import RandomFunctionSchedulerFactory
from ext.jjnp21.automator.factories.lb_scaler import FractionLoadBalancerScalerFactory
from ext.jjnp21.automator.factories.lb_scheduler import EverywhereLoadBalancerSchedulerFactory
from ext.jjnp21.automator.factories.topology import GlobalDistributedUrbanSensingFactory, \
    GlobalDistributedRealisticCityFactory
from ext.jjnp21.automator.main import ExperimentRunAutomator

logging.basicConfig(level=logging.INFO)
rps = 50
duration = 750


def get_experiment_for_fraction(fraction: float) -> Experiment:
    return Experiment(f'LRT {fraction}',
                      seed=13,
                      lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                      lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                      client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                      client_placement_strategy=ClientPlacementStrategy.NONE,
                      benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                      faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                      net_mode=NetworkSimulationMode.ACCURATE,
                      function_scheduler_factory=RandomFunctionSchedulerFactory(),
                      lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=fraction),
                      lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
                      # topology_factory=GlobalDistributedRealisticCityFactory(seed=13, client_ratio=0.6))
                      topology_factory=GlobalDistributedUrbanSensingFactory(client_ratio=0.6))


# end of experiments

# start = time.time()
# result = run_experiment(experiment)
# end = time.time()
# print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')
# analyzer = BasicResultAnalyzer([result])
# analysis_df = analyzer.basic_kpis()
# md = analysis_df.to_markdown()
# print(md)

# start = time.time()
# result = run_experiment(e4)
# end = time.time()
# print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')
# analyzer = BasicResultAnalyzer([result])
# analysis_df = analyzer.basic_kpis()
# md = analysis_df.to_markdown()
# print(md)
# exit(0)


# experiment_list = [get_experiment_for_fraction(f) for f in [0.1, 0.2]]
experiment_list = [get_experiment_for_fraction(f) for f in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
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

analysis_df.to_csv('/home/jp/Documents/tmp/analysis.csv', sep=';')
print('successfully ran analysis')
print('dumping results')
f = open('/home/jp/Documents/tmp/results.dump', 'wb')
pickle.dump(results, f)
f.flush()
f.close()
print('successfully dumped results')
