import logging
from typing import List

import ether.scenarios.urbansensing as scenario
from faas.system.core import FunctionImage, FunctionRequest, FunctionContainer, Function
from skippy.core.utils import parse_size_string

from sim import docker
from sim.benchmark import Benchmark
from sim.context.platform.deployment.model import SimFunctionDeployment, SimScalingConfiguration, DeploymentRanking
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas.core import SimResourceConfiguration
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
            ps.append(env.process(env.faas.invoke(FunctionRequest('python-pi', env.now))))

        logger.info('executing 10 resnet50-inference requests')
        for i in range(10):
            ps.append(env.process(env.faas.invoke(FunctionRequest('resnet50-inference', env.now))))

        # wait for invocation processes to finish
        for p in ps:
            yield p

    def prepare_deployments(self) -> List[SimFunctionDeployment]:
        resnet_fd = self.prepare_resnet_inference_deployment()

        python_pi_fd = self.prepare_python_pi_deployment()

        return [python_pi_fd, resnet_fd]

    def prepare_python_pi_deployment(self):
        # Design Time

        python_pi = 'python-pi'
        python_pi_cpu = FunctionImage(image='python-pi-cpu')
        python_pi_fn = Function(python_pi, fn_images=[python_pi_cpu])

        # Run time

        python_pi_fn_container = FunctionContainer(python_pi_cpu, SimResourceConfiguration())

        python_pi_fd = SimFunctionDeployment(
            python_pi_fn,
            [python_pi_fn_container],
            SimScalingConfiguration()
        )

        return python_pi_fd

    def prepare_resnet_inference_deployment(self):
        # Design time

        resnet_inference = 'resnet50-inference'
        inference_cpu = 'resnet50-inference-cpu'
        inference_gpu = 'resnet50-inference-gpu'

        resnet_inference_gpu = FunctionImage(image=inference_gpu)
        resnet_inference_cpu = FunctionImage(image=inference_cpu)
        resnet_fn = Function(resnet_inference, fn_images=[resnet_inference_gpu, resnet_inference_cpu])

        # Run time

        # default kubernetes requested resources
        resnet_cpu_container = FunctionContainer(resnet_inference_cpu, SimResourceConfiguration())

        # custom defined requested resources
        request = SimResourceConfiguration.create_from_str(cpu='100m', memory='1024Mi')
        resnet_gpu_container = FunctionContainer(resnet_inference_gpu, resource_config=request)

        resnet_fd = SimFunctionDeployment(
            resnet_fn,
            [resnet_cpu_container, resnet_gpu_container],
            SimScalingConfiguration(),
            DeploymentRanking([resnet_gpu_container, resnet_cpu_container])
        )

        return resnet_fd


if __name__ == '__main__':
    main()
