from typing import Dict

from ext.jjnp21.benchmarks.constant_lb import LoadBalancerConstant
from ext.raith21.benchmark.constant import ConstantBenchmark
from sim.benchmark import Benchmark


# Factory is there to enable a stronger decoupling of the benchmarks used in experiments, while keeping the
# "Experiment" instances themselves clean and purely as a type of dataclass without its own behaviour
class BenchmarkFactory:
    def create(self) -> Benchmark:
        raise Exception('This is just a placeholder class. Use an actual implementation instead!')


class ConstantBenchmarkFactory(BenchmarkFactory):
    requests_per_second: int = 50
    duration: int = 300

    def __init__(self, requests_per_second: int = 50, duration: int = 300) -> None:
        self.requests_per_second = requests_per_second
        self.duration = duration

    def create(self) -> Benchmark:
        return ConstantBenchmark('mixed', duration=self.duration, rps=self.requests_per_second)


class LBConstantBenchmarkFactory(ConstantBenchmarkFactory):
    kwargs: Dict = {}

    def __init__(self, requests_per_second: int = 50, duration: int = 300, lb_profile: str = 'LRT') -> None:
        self.lb_profile = lb_profile
        super().__init__(requests_per_second, duration)

    def create(self) -> Benchmark:
        return LoadBalancerConstant('mixed', duration=self.duration, rps=self.requests_per_second,
                                    lb_profile=self.lb_profile)
