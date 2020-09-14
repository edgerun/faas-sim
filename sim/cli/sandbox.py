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

        containers.put(ImageProperties('edgerun/ml_workflow/preprocess', parse_size_string('20M')))
        containers.put(ImageProperties('edgerun/ml_workflow/train', parse_size_string('300M')))

        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                print(name, tag, images)

    def run(self, env: Environment):
        yield from env.faas.deploy(FunctionDefinition('wf_01_pre', 'edgerun/ml_workflow/preprocess'))

        # execute 10 requests and wait 1 second between each request
        for i in range(10):
            print('requesting')
            yield env.timeout(1)
            yield from env.faas.request(FunctionRequest('wf_01_pre'))


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
