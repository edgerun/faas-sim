import logging

import ether.scenarios.urbansensing as scenario

from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.metrics import Metrics
from sim.topology import Topology, DockerRegistry
from skippy.core.utils import parse_size_string


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
    t.init_docker_registry()

    return t


def main():
    # TODO: read experiment specification
    topology = example_topology()

    logging.basicConfig(level=logging.DEBUG)
    env = Environment()
    env.topology = topology
    env.metrics = Metrics(env)
    env.registry = docker.Registry()
    env.registry.put(ImageProperties('edgerun/go-telemd', size=parse_size_string('13M')))

    route = topology.route(DockerRegistry, topology.find_node('rpi3_0'))
    print(route)
    flow = docker.pull(env, 'edgerun/go-telemd', topology.find_node('rpi3_0'))

    env.process(flow)
    env.run()
    print(env.now)

    # exp = faassim.Simulation(env, ExampleBenchmark(), topology)
    # exp.run()


if __name__ == '__main__':
    main()
