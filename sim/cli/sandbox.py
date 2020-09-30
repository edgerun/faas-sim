import logging

import ether.scenarios.urbansensing as scenario

from faassim.faas import FunctionRequest
from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionDefinition
from sim.faassim import Simulation
from sim.topology import Topology
from skippy.core.utils import parse_size_string


class ExampleBenchmark(Benchmark):

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        containers.put(ImageProperties('smttest', parse_size_string('58M')))
        containers.put(ImageProperties('python-pi', parse_size_string('56M')))

        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                print(name, tag, images)

    def run(self, env: Environment):
        yield from env.faas.deploy(FunctionDefinition('smttest', 'smttest'))

        # execute 10 requests and wait 1 second between each request
        for i in range(10):
            yield env.timeout(1)
            yield from env.faas.invoke(FunctionRequest('smttest'))


def example_topology() -> Topology:
    t = Topology()
    scenario.UrbanSensingScenario().materialize(t)
    t.init_docker_registry()

    return t


def main():
    logging.basicConfig(level=logging.DEBUG)

    # TODO: read experiment specification
    topology = example_topology()
    benchmark = ExampleBenchmark()

    sim = Simulation(topology, benchmark)
    sim.run()


if __name__ == '__main__':
    main()
