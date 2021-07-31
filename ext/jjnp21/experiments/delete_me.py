import time
from ext.jjnp21.automator.analyzer import BasicResultAnalyzer
from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import ConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import LocalizedLoadBalancerFaaSFactory
from ext.jjnp21.automator.factories.topology import RaithHeterogeneousUrbanSensingFactory, GlobalIndustrialIoTScenario, NationDistributedUrbanSensingFactory
from ext.jjnp21.automator.main import ExperimentRunAutomator
from ext.jjnp21.automator.experiment import Experiment, LoadBalancerType
import logging

logging.basicConfig(level=logging.INFO)
rps = 75
duration = 500


rrc = Experiment('Round Robin centralized',
                lb_type=LoadBalancerType.ROUND_ROBIN,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                function_scaling_strategy = FunctionScalingStrategy.AVG_QUEUE_LENGTH,
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))
rrd = Experiment('Round Robin on all nodes',
                lb_type=LoadBalancerType.ROUND_ROBIN,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                function_scaling_strategy = FunctionScalingStrategy.AVG_QUEUE_LENGTH,
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))
lrtc = Experiment('Least Response Time centralized',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                function_scaling_strategy = FunctionScalingStrategy.AVG_QUEUE_LENGTH,
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))
lrtd = Experiment('Least Response Time on all nodes',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(rps, duration),
                faas_system_factory=LocalizedLoadBalancerFaaSFactory(),
                function_scaling_strategy = FunctionScalingStrategy.AVG_QUEUE_LENGTH,
                topology_factory=RaithHeterogeneousUrbanSensingFactory(client_ratio=0.6))

experiment_list = [rrc, rrd, lrtc, lrtd]
automator = ExperimentRunAutomator(experiment_list, worker_count=4)
start = time.time()
results = automator.run()
end = time.time()
print(f'Done calculating... E2E runtime: {round(end - start, 2)}s')
# results.sort('experiment.name')
for r in results:
    print(f'Ran "{r.experiment.name}" in {r.run_duration_seconds}s')
