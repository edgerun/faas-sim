import logging

from ext.jjnp21.automator.execution import run_experiment
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import ConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import LocalizedLoadBalancerFaaSFactory
from ext.jjnp21.automator.factories.function_scheduler import RandomFunctionSchedulerFactory
from ext.jjnp21.automator.factories.lb_scaler import FractionLoadBalancerScalerFactory
from ext.jjnp21.automator.factories.lb_scheduler import EverywhereLoadBalancerSchedulerFactory
from ext.jjnp21.debugging.tiny_topology_factory import TinyUrbanSensingTopologyFactory
from ext.jjnp21.scalers.fraction_lb_scaler import FractionScaler

logging.basicConfig(level=logging.DEBUG)
node_count = 100
rps = 25
duration = 500
experiment = Experiment('Least Response Time on all nodes',
                        lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                        lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                        client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                        client_placement_strategy=ClientPlacementStrategy.NONE,
                        benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                        faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                        net_mode=NetworkSimulationMode.FAST,
                        function_scheduler_factory=RandomFunctionSchedulerFactory(),
                        topology_factory=TinyUrbanSensingTopologyFactory(client_ratio=1, node_count=20),
                        lb_scaler_facotry=FractionLoadBalancerScalerFactory(target_fraction=1.0),
                        lb_scheduler_factory=EverywhereLoadBalancerSchedulerFactory())


run_experiment(experiment)

# # experiment_list = [e2]
# experiment_list = [e1, e2, e3, e4]
# # result = run_experiment(e1)
#
# automator = ExperimentRunAutomator(experiment_list, worker_count=4)
# print('Running nation benchmark')
# start = time.time()
# results = automator.run()
# end = time.time()
# print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')
# # results.sort('experiment.name')
# for r in results:
#     print(f'Ran "{r.experiment.name}" in {r.run_duration_seconds}s')
#
# analyzer = BasicResultAnalyzer(results)
# analysis_df = analyzer.basic_kpis()
# analysis_df.to_csv('/home/jp/Documents/tmp/analysis.csv', sep=';')
# print('successfully ran analysis')
# print('dumping results')
# f = open('/home/jp/Documents/tmp/results.dump', 'wb')
# pickle.dump(results, f)
# f.flush()
# f.close()
# print('successfully dumped results')
