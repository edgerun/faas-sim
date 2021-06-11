from ext.jjnp21.automator.experiment import *
from ext.jjnp21.automator.factories.benchmark import ConstantBenchmarkFactory
from ext.jjnp21.automator.factories.faas import LocalizedLoadBalancerFaaSFactory
from ext.jjnp21.automator.factories.topology import RaithHeterogeneousUrbanSensingFactory
from ext.jjnp21.automator.main import ExperimentRunAutomator

e1 = Experiment('Round Robin centralized',
                lb_type=LoadBalancerType.ROUND_ROBIN,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(10, 111),
                faas_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(100))
e2 = Experiment('Round Robin on all nodes',
                lb_type=LoadBalancerType.ROUND_ROBIN,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(10, 111),
                faas_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(100))
e3 = Experiment('Least Response Time centralized',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.CENTRAL,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(10, 111),
                faas_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(100))
e4 = Experiment('Least Response Time on all nodes',
                lb_type=LoadBalancerType.LEAST_RESPONSE_TIME,
                lb_placement_strategy=LoadBalancerPlacementStrategy.ALL_NODES,
                client_lb_resolving_strategy=ClientLoadBalancerResolvingStrategy.LOWEST_PING,
                client_placement_strategy=ClientPlacementStrategy.NONE,
                benchmark_factory=ConstantBenchmarkFactory(10, 111),
                faas_factory=LocalizedLoadBalancerFaaSFactory(),
                topology_factory=RaithHeterogeneousUrbanSensingFactory(100))

experiment_list = [e1, e2, e3, e4]

automator = ExperimentRunAutomator(experiment_list)
results = automator.run()
for result in results:
    print(result.experiment.name)
    print(result.experiment.duration)

