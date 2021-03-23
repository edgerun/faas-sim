import logging

import ether.scenarios.urbansensing as scenario
from skippy.core.utils import parse_size_string

from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionDefinition, FunctionDeployment, FunctionRequest
from sim.faassim import Simulation
from sim.topology import Topology

logger = logging.getLogger(__name__)


class ExampleBenchmark(Benchmark):

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        # fill global container registry with images
        containers.put(ImageProperties('smttest-cpu', parse_size_string('58M'), arch='arm32'))
        containers.put(ImageProperties('smttest-cpu', parse_size_string('58M'), arch='x86'))
        containers.put(ImageProperties('smttest-cpu', parse_size_string('58M'), arch='aarch64'))

        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='aarch64'))

        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='aarch64'))

        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        deployments = self.prepare_deployments()

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

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

    def prepare_deployments(self):
        smttest_deployment = FunctionDeployment(
            name='smttest',
            function_definitions={
                'smttest-cpu': FunctionDefinition(name='smttest', image='smttest-cpu')
            }
        )
        resnet50_inference_deployment = FunctionDeployment(
            name='resnet50-inference',
            function_definitions={
                'resnet50-inference-gpu': FunctionDefinition(name='resnet50-inference', image='resnet50-inference-gpu'),
                'resnet50-inference-cpu': FunctionDefinition(name='resnet50-inference', image='resnet50-inference-cpu')
            }
        )

        return [smttest_deployment, resnet50_inference_deployment]


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
