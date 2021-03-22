import logging

import ether.scenarios.urbansensing as scenario
from skippy.core.utils import parse_size_string

from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionDefinition
from sim.faas import FunctionRequest
from sim.faassim import Simulation
from sim.topology import Topology

logger = logging.getLogger(__name__)


class ExampleBenchmark(Benchmark):

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        # fill global container registry with images
        containers.put(ImageProperties('smttest', parse_size_string('58M')))
        containers.put(ImageProperties('python-pi', parse_size_string('56M')))

        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        yield from env.faas.deploy(FunctionDefinition('smttest', 'smttest'))

        logger.info('waiting for replica')
        yield env.process(env.faas.poll_available_replica('smttest'))

        # execute 10 requests in parallel
        logger.info('executing requests')
        ps = []
        for i in range(16):
            ps.append(env.process(env.faas.invoke(FunctionRequest('smttest'))))

        # wait for invocation processes to finish
        for p in ps:
            yield p


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
