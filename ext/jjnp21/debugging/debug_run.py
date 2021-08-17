import logging
import pickle

from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.execution import run_experiment
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import ConstantBenchmarkFactory, LBConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import LocalizedLoadBalancerFaaSFactory
from ext.jjnp21.automator.factories.function_scheduler import RandomFunctionSchedulerFactory
from ext.jjnp21.automator.factories.lb_scaler import FractionLoadBalancerScalerFactory, \
    ManualSetLoadBalancerScalerFactory
from ext.jjnp21.automator.factories.lb_scheduler import EverywhereLoadBalancerSchedulerFactory, \
    CentralLoadBalancerSchedulerFactory
from ext.jjnp21.automator.factories.topology import GlobalDistributedUrbanSensingFactory
from ext.jjnp21.automator.main import ExperimentRunAutomator
from ext.jjnp21.debugging.tiny_topology_factory import TinyUrbanSensingTopologyFactory
from ext.jjnp21.scalers.fraction_lb_scaler import FractionScaler
import time

logging.basicConfig(level=logging.INFO)
rps = 25
duration = 500
experiment = Experiment('Least Response Time on all nodes',
                        lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                        lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                        client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                        client_placement_strategy=ClientPlacementStrategy.NONE,
                        benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                        faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                        net_mode=NetworkSimulationMode.FAST,
                        function_scheduler_factory=RandomFunctionSchedulerFactory(),
                        topology_factory=TinyUrbanSensingTopologyFactory(client_ratio=1, node_count=20),
                        lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=1.0),
                        lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory())

# beginning of experiments
e3 = Experiment('Least Response Time centralized',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.FAST,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=ManualSetLoadBalancerScalerFactory(target_count=1),
                lb_scheduler_factory=CentralLoadBalancerSchedulerFactory(),
                topology_factory=GlobalDistributedUrbanSensingFactory(client_ratio=0.6))

e4 = Experiment('Least Response Time on all nodes',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'LRT'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.FAST,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=1.0),
                lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
                topology_factory=GlobalDistributedUrbanSensingFactory(client_ratio=0.6))

e1 = Experiment('Round Robin centralized',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'RR'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.FAST,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=ManualSetLoadBalancerScalerFactory(target_count=1),
                lb_scheduler_factory=CentralLoadBalancerSchedulerFactory(),
                topology_factory=GlobalDistributedUrbanSensingFactory(client_ratio=0.6))

e2 = Experiment('Round Robin on all nodes',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=LBConstantBenchmarkFactory(rps, duration, 'RR'),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                net_mode=NetworkSimulationMode.FAST,
                function_scheduler_factory=RandomFunctionSchedulerFactory(),
                lb_scaler_factory=FractionLoadBalancerScalerFactory(target_fraction=1.0),
                lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory(),
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

start = time.time()
result = run_experiment(e3)
end = time.time()
print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')
analyzer = BasicResultAnalyzer([result])
analysis_df = analyzer.basic_kpis()
md = analysis_df.to_markdown()
print(md)

experiment_list = [e1, e2, e3, e4]
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
