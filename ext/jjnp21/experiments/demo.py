import time
from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import ConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import LocalizedLoadBalancerFaaSFactory
from ext.jjnp21.automator.factories.topology import RaithHeterogeneousUrbanSensingFactory, GlobalIndustrialIoTScenario
from ext.jjnp21.automator.main import ExperimentRunAutomator

node_count = 100
rps = 50
duration = 500

e1 = Experiment('Round Robin centralized',
                lb_type=LoadBalancerType.ROUND_ROBIN,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))
e2 = Experiment('Round Robin on all nodes',
                lb_type=LoadBalancerType.ROUND_ROBIN,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))
e3 = Experiment('Least Response Time centralized',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))
e4 = Experiment('Least Response Time on all nodes',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))

experiment_list = [e1, e2, e3, e4]

automator = ExperimentRunAutomator(experiment_list, worker_count=4)
start = time.time()
results = automator.run()
end = time.time()
print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')
# results.sort('experiment.name')
for r in results:
    print(f'Ran "{r.experiment.name}" in {r.run_duration_seconds}s')
analyzer = BasicResultAnalyzer(results)
analysis_df = analyzer.basic_kpis()
analysis_df.to_csv('/home/jp/Documents/tmp/analysis.csv', sep=';')
print('successfully ran analysis')


