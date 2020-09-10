from ether.topology import Topology

from sim.core import Environment


class Benchmark:
    # the benchmark contains generators for users, functions, initial deployments, deployments over time,
    # requests over time

    def setup(self, env: Environment):
        pass

    def run(self, env: Environment):
        yield env.timeout(0)
