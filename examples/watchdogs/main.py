import logging
from typing import List

from ether.util import parse_size_string

import examples.basic.main as basic
from examples.watchdogs.inference import InferenceFunctionSim
from examples.watchdogs.training import TrainingFunctionSim
from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import SimulatorFactory, FunctionSimulator, SimFunctionDeployment, SimScalingConfiguration, \
    DeploymentRanking
from sim.faas.core import SimResourceConfiguration
from sim.faassim import Simulation
from faas.system.core import FunctionContainer, FunctionRequest, FunctionImage, Function

logger = logging.getLogger(__name__)


class AIFunctionSimulatorFactory(SimulatorFactory):

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        if 'inference' in fn.fn_image.image:
            return InferenceFunctionSim(4)
        elif 'training' in fn.fn_image.image:
            return TrainingFunctionSim()


def main():
    logging.basicConfig(level=logging.INFO)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(basic.example_topology(), TrainInferenceBenchmark())

    # override the SimulatorFactory factory
    sim.create_simulator_factory = AIFunctionSimulatorFactory

    # run the simulation
    sim.run()
    dfs = {
        'fets_df': sim.env.metrics.extract_dataframe('fets')
    }
    pass


class TrainInferenceBenchmark(Benchmark):

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='aarch64'))

        containers.put(ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='x86'))
        containers.put(ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='aarch64'))

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
        yield env.process(env.faas.poll_available_replica('resnet50-training'))
        yield env.process(env.faas.poll_available_replica('resnet50-inference'))

        # run workload
        ps = []
        # execute 10 requests in parallel
        logger.info('executing 10 resnet50-training requests')
        for i in range(10):
            ps.append(env.process(env.faas.invoke(FunctionRequest('resnet50-training', env.now))))

        logger.info('executing 10 resnet50-inference requests')
        for i in range(10):
            ps.append(env.process(env.faas.invoke(FunctionRequest('resnet50-inference', env.now))))

        # wait for invocation processes to finish
        for p in ps:
            yield p

    def prepare_deployments(self) -> List[SimFunctionDeployment]:
        resnet_inference_fd = self.prepare_resnet_inference_deployment()

        resnet_training_fd = self.prepare_resnet_training_deployment()

        return [resnet_training_fd, resnet_inference_fd]

    def prepare_resnet_training_deployment(self):
        # Design time

        resnet_training = 'resnet50-training'
        training_cpu = 'resnet50-training-cpu'

        resnet_training_cpu = FunctionImage(image=training_cpu)
        resnet_fn = Function(resnet_training, fn_images=[resnet_training_cpu])

        # Run time

        resnet_cpu_container = FunctionContainer(resnet_training_cpu, SimResourceConfiguration())

        resnet_fd = SimFunctionDeployment(
            resnet_fn,
            [resnet_cpu_container],
            SimScalingConfiguration(),
            DeploymentRanking([training_cpu])
        )

        return resnet_fd

    def prepare_resnet_inference_deployment(self):
        # Design time

        resnet_inference = 'resnet50-inference'
        inference_cpu = 'resnet50-inference-cpu'

        resnet_inference_cpu = FunctionImage(image=inference_cpu)
        resnet_fn = Function(resnet_inference, fn_images=[resnet_inference_cpu])

        # Run time

        resnet_cpu_container = FunctionContainer(resnet_inference_cpu, SimResourceConfiguration())

        resnet_fd = SimFunctionDeployment(
            resnet_fn,
            [resnet_cpu_container],
            SimScalingConfiguration(),
            DeploymentRanking([inference_cpu])
        )

        return resnet_fd


if __name__ == '__main__':
    main()
