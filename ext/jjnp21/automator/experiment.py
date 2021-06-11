from dataclasses import dataclass
from enum import Enum
from pandas import DataFrame

from ext.jjnp21.automator.factories.benchmark import BenchmarkFactory
from ext.jjnp21.automator.factories.faas import FaaSFactory
from ext.jjnp21.automator.factories.topology import TopologyFactory


class LoadBalancerType(Enum):
    ROUND_ROBIN = 1
    LEAST_RESPONSE_TIME = 2


class LoadBalancerPlacementStrategy(Enum):
    CENTRAL = 1
    ALL_NODES = 1


class ClientLoadBalancerResolvingStrategy(Enum):
    LOWEST_PING = 1


class ClientPlacementStrategy(Enum):
    NONE = 1
    UNIFORM_RANDOM = 2


@dataclass(init=True)
class Experiment:
    name: str
    topology_factory: TopologyFactory
    benchmark_factory: BenchmarkFactory
    faas_factory: FaaSFactory
    lb_type: LoadBalancerType
    lb_placement_strategy: LoadBalancerPlacementStrategy
    client_lb_resolving_strategy: ClientLoadBalancerResolvingStrategy
    client_placement_strategy: ClientPlacementStrategy
    duration: int = 300
    req_per_sec: int = 50
    seed: int = 42


@dataclass()
class Result:
    experiment: Experiment
    run_duration_seconds: float
    invocations: DataFrame
    scale: DataFrame
    schedule: DataFrame
    replica_deployment: DataFrame
    function_deployment: DataFrame
    function_deployments: DataFrame
    function_deployment_lifecycle: DataFrame
    functions: DataFrame
    flow: DataFrame
    network: DataFrame
    utilization: DataFrame
    fets: DataFrame

    def __init__(self):
        pass


