import logging
import time
from typing import List

from faas.system import FunctionContainer
from skippy.core.scheduler import Scheduler

from examples.decentralized_clients.clients import ClientSimulator
from examples.decentralized_clients.deployments import get_resnet50_inference_cpu_image_properties, \
    get_resnet50_training_cpu_image_properties, get_galileo_worker_image_properties, prepare_function_deployments, \
    prepare_training_clients, prepare_inference_clients
from examples.decentralized_loadbalancers.topology import testbed_topology
from examples.watchdogs.inference import InferenceFunctionSim
from examples.watchdogs.training import TrainingFunctionSim
from sim import docker
from sim.benchmark import Benchmark
from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.faas import FunctionSimulator, SimulatorFactory
from sim.faas.core import Node
from sim.faassim import Simulation
from sim.predicates import PodHostEqualsNode
from sim.requestgen import FunctionRequestFactory, SimpleFunctionRequestFactory
from sim.util.client import find_clients
from sim.util.experiment import extract_dfs

logger = logging.getLogger(__name__)

"""
This example shows how to generate requests from different nodes.
Specifically, we want to include the network latency between client nodes (up to users to select)
and function hosting nodes.
This is done via scheduling Pods that simulate client behavior (i.e., invoking a function)
This behavior is implemented in the ClientSimulator, which we include in the SimulatorFactory.
This lets us create FunctionDeployments that, when invoked, will generate requests. 
Note, that while the requests are made from the clients, the load balancing decision will still happen centralized.
"""


def create_scheduler(env: Environment):
    scheduler = Scheduler(env.cluster)
    scheduler.predicates.append(PodHostEqualsNode())
    # scheduler.predicates.append(CheckNodeLabelPresencePred([client_label]))
    return scheduler


class ClientAIFunctionSimulatorFactory(SimulatorFactory):

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        if 'inference' in fn.fn_image.image:
            return InferenceFunctionSim(4)
        elif 'training' in fn.fn_image.image:
            return TrainingFunctionSim()
        elif 'galileo-worker' in fn.fn_image.image:
            return ClientSimulator()


class DecentralizedTrainInferenceBenchmark(Benchmark):

    def __init__(self, clients: List[Node], inference_request_factory: FunctionRequestFactory,
                 training_request_factory: FunctionRequestFactory):
        self.clients = clients
        self.inference_request_factory = inference_request_factory
        self.training_request_factory = training_request_factory

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        images = []
        resnet50_inference_cpu_img_properties = get_resnet50_inference_cpu_image_properties()
        resnet50_training_cpu_img_properties = get_resnet50_training_cpu_image_properties()
        galileo_worker_image_properties = get_galileo_worker_image_properties()

        images.extend(resnet50_inference_cpu_img_properties)
        images.extend(resnet50_training_cpu_img_properties)
        images.extend(galileo_worker_image_properties)

        for image in images:
            containers.put(image)

        # log all the images in the container
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        # deploy functions
        deployments = []
        fn_deployments = prepare_function_deployments()
        client_deployments = self.prepare_client_deployments_for_experiment(fn_deployments)

        deployments.extend(fn_deployments)
        deployments.extend(client_deployments)

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        # block until replicas become available (scheduling has finished and replicas have been deployed on the node)
        logger.info('waiting for replicas')
        for deployment in deployments:
            yield env.process(env.faas.poll_available_replica(deployment.name))

        # run workload
        ps = []
        request_factory = SimpleFunctionRequestFactory()
        for deployment in client_deployments:
            ps.append(env.process(env.faas.invoke(request_factory.generate(env, deployment))))

        # wait for invocation processes to finish
        for p in ps:
            yield p

    def prepare_client_deployments_for_experiment(self, deployments: List[SimFunctionDeployment]) -> List[
        SimFunctionDeployment]:

        fds = []
        inference_clients = [self.clients[0].name]
        training_clients = [self.clients[1].name]
        fds.extend(prepare_inference_clients(inference_clients, self.inference_request_factory, deployments[0]))
        fds.extend(prepare_training_clients(training_clients, self.training_request_factory, deployments[1]))

        return fds


def execute_benchmark():
    topology = testbed_topology()

    clients = find_clients(topology)

    #  inference factory - we assume that the file size is 250KB - client is a raspberry pi
    inference_factory = SimpleFunctionRequestFactory(size=250)

    # training factory - we assume that the file size is 10MB (100000KB) - client is a raspberry pi
    train_factory = SimpleFunctionRequestFactory(size=10000)

    benchmark = DecentralizedTrainInferenceBenchmark(clients, inference_factory, train_factory)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(topology, benchmark)
    sim.create_simulator_factory = ClientAIFunctionSimulatorFactory

    # override the SimulatorFactory factory
    sim.env.benchmark = benchmark
    sim.env.topology = topology
    sim.init_environment(sim.env)
    sim.env.scheduler = create_scheduler(sim.env)

    # run the simulation
    start = time.time()
    sim.run()
    end = time.time()
    duration = end - start

    return duration, sim


def main():
    logging.basicConfig(level=logging.DEBUG)
    duration, sim = execute_benchmark()
    env = sim.env
    dfs = extract_dfs(sim)
    logger.info(f'Time passed in simulation: {env.now}, wall time passed: {duration}')
    logger.info('Mean exec time %d', dfs['invocations_df']['ts_exec'].mean())
    logger.info(f'Fets invocations: {len(dfs["fets_df"])}')


if __name__ == '__main__':
    main()
