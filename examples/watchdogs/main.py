import logging
from typing import List

from ether.util import parse_size_string
from faas.system.core import FunctionContainer, FunctionImage, Function

import examples.basic.main as basic
from examples.watchdogs.inference import InferenceFunctionSim
from examples.watchdogs.training import TrainingFunctionSim
from sim import docker
from sim.benchmark import Benchmark
from sim.context.platform.deployment.model import SimFunctionDeployment, SimScalingConfiguration, DeploymentRanking
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import SimulatorFactory, FunctionSimulator
from sim.faas.core import SimResourceConfiguration
from sim.faassim import Simulation
from sim.requestgen import expovariate_arrival_profile, constant_rps_profile, function_trigger, FunctionRequestFactory, \
    SimpleFunctionRequestFactory

logger = logging.getLogger(__name__)


class AIFunctionSimulatorFactory(SimulatorFactory):

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        if 'inference' in fn.fn_image.image:
            return InferenceFunctionSim(4)
        elif 'training' in fn.fn_image.image:
            return TrainingFunctionSim()


def main():
    logging.basicConfig(level=logging.INFO)

    #  inference factory - we assume that the file size is 250KB - client is a raspberry pi
    inference_factory = SimpleFunctionRequestFactory(client='rpi3_0', size=250)

    # training factory - we assume that the file size is 10MB (100000KB) - client is a raspberry pi
    train_factory = SimpleFunctionRequestFactory(client='rpi3_1', size=10000)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(basic.example_topology(), TrainInferenceBenchmark(inference_factory, train_factory))

    # override the SimulatorFactory factory
    sim.create_simulator_factory = AIFunctionSimulatorFactory

    # run the simulation
    sim.run()

    dfs = {
        'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
        'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
        'scale_df': sim.env.metrics.extract_dataframe('scale'),
        'schedule_df': sim.env.metrics.extract_dataframe('schedule'),
        'replica_deployment_df': sim.env.metrics.extract_dataframe('replica_deployment'),
        'function_deployments_df': sim.env.metrics.extract_dataframe('function_deployments'),
        'function_deployment_df': sim.env.metrics.extract_dataframe('function_deployment'),
        'function_deployment_lifecycle_df': sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
        'functions_df': sim.env.metrics.extract_dataframe('functions'),
        'flow_df': sim.env.metrics.extract_dataframe('flow'),
        'network_df': sim.env.metrics.extract_dataframe('network'),
        'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
        'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization'),
        'fets_df': sim.env.metrics.extract_dataframe('fets')
    }
    pass


class TrainInferenceBenchmark(Benchmark):

    def __init__(self, inference_request_factory: FunctionRequestFactory,
                 training_request_factory: FunctionRequestFactory):
        self.inference_request_factory = inference_request_factory
        self.training_request_factory = training_request_factory

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

        max_rps_inference = 5
        max_requests_inference = 400
        logger.info(
            f'executing resnet50-inference requests with {max_rps_inference} rps and maximum {max_requests_inference}')
        # generate profile
        ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=max_rps_inference), max_ia=1)

        # run profile
        ps.append(env.process(
            function_trigger(env, deployments[1], self.inference_request_factory, ia_generator,
                             max_requests=max_requests_inference)))

        max_rps_training = 1
        max_requests_training = 10
        logger.info(
            f'executing resnet50-training requests with {max_rps_training} rps and maximum {max_requests_inference}')
        # generate profile
        ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=max_rps_training), max_ia=1)

        # run profile
        ps.append(env.process(
            function_trigger(env, deployments[0], self.training_request_factory, ia_generator,
                             max_requests=max_requests_training)))

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
