import logging
from typing import List

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


def main():
    logging.basicConfig(level=logging.DEBUG)

    # a topology holds the cluster configuration and network topology
    topology = example_topology()

    # a benchmark is a simpy process that sets up the runtime system (e.g., creates container images, deploys functions)
    # and creates workload by simulating function requests
    benchmark = ExampleBenchmark()

    # a simulation runs until the benchmark process terminates
    sim = Simulation(topology, benchmark)
    sim.run()


def example_topology() -> Topology:
    t = Topology()
    scenario.UrbanSensingScenario().materialize(t)
    t.init_docker_registry()

    return t


class ExampleBenchmark(Benchmark):

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        # populate the global container registry with images
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='arm32'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='x86'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='aarch64'))

        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='aarch64'))

        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='aarch64'))

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        # deploy functions
        deployments = self.prepare_deployments()

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        # block until replicas become available (scheduling has finished and replicas have been deployed on the node)
        logger.info('waiting for replica')
        yield env.process(env.faas.poll_available_replica('python-pi'))
        yield env.process(env.faas.poll_available_replica('resnet50-inference'))

        # run workload
        ps = []
        # execute 10 requests in parallel
        logger.info('executing 10 python-pi requests')
        for i in range(10):
            ps.append(env.process(env.faas.invoke(FunctionRequest('python-pi'))))

        logger.info('executing 10 resnet50-inference requests')
        for i in range(10):
            ps.append(env.process(env.faas.invoke(FunctionRequest('resnet50-inference'))))

        # wait for invocation processes to finish
        for p in ps:
            yield p

    def prepare_deployments(self) -> List[FunctionDeployment]:
        python_pi_deployment = FunctionDeployment(
            name='python-pi',
            function_definitions={
                'python-pi-cpu': FunctionDefinition(name='python-pi', image='python-pi-cpu')
            }
        )
        resnet50_inference_deployment = FunctionDeployment(
            name='resnet50-inference',
            function_definitions={
                'resnet50-inference-gpu': FunctionDefinition(name='resnet50-inference', image='resnet50-inference-gpu'),
                'resnet50-inference-cpu': FunctionDefinition(name='resnet50-inference', image='resnet50-inference-cpu')
            }
        )

        return [python_pi_deployment, resnet50_inference_deployment]


if __name__ == '__main__':
    main()
