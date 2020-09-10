import logging

import ether.scenarios.urbansensing as scenario

import sim.faassim as faassim
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.topology import Topology


class ExampleBenchmark(Benchmark):

    def run(self, env: Environment):
        yield env.faas.deploy('my_image_classifier')

        ## execute 1000 requests and wait 1 second between each request
        for i in range(1000):
            yield env.timeout(1)
            yield env.faas.request('my_image_classifier', {})


def example_topology() -> Topology:
    t = Topology()
    scenario.UrbanSensingScenario().materialize(t)
    return t


def main():
    # TODO: read experiment specification

    logging.basicConfig(level=logging.DEBUG)
    env = Environment()
    exp = faassim.Simulation(env, ExampleBenchmark(), example_topology())

    exp.run()


if __name__ == '__main__':
    main()
