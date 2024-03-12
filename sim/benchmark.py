import logging
import os
from typing import List, Dict, Generator

from ext.raith21 import loader
from sim import docker
from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.docker import ImageProperties
from sim.requestgen import function_trigger, FunctionRequestFactory

logger = logging.getLogger(__name__)

class Benchmark:
    # the benchmark contains generators for users, functions, initial deployments, deployments over time,
    # requests over time

    # this metadata object can be filled with experiment settings etc. and will be saved with the experiment
    metadata: Dict = {}

    def setup(self, env: Environment):
        pass

    def run(self, env: Environment):
        yield env.timeout(0)


class BenchmarkBase(Benchmark):
    def __init__(self, images: List[ImageProperties], deployments: List[SimFunctionDeployment],
                 arrival_profiles: Dict[str, Generator], fn_request_factories: Dict[str, FunctionRequestFactory],
                 duration: int = None):
        self.duration = duration  # in seconds
        self.images = images
        self.deployments = deployments
        self.deployments_per_name = self.__create_deployments_per_name()
        self.fn_request_factories = fn_request_factories
        self.arrival_profiles = arrival_profiles

    def __create_deployments_per_name(self):
        deployments_per_name = {}
        for deployment in self.deployments:
            deployments_per_name[deployment.name] = deployment
        return deployments_per_name

    def setup(self, env: Environment):
        super().setup(env)
        self.register_images(env)

    def register_images(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        if len(containers.images) == 0:
            for properties in self.images:
                containers.put(properties)

            for name, tag_dict in containers.images.items():
                for tag, images in tag_dict.items():
                    logging.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        for deployment in self.deployments:
            yield from env.faas.deploy(deployment)
        for deployment in self.deployments:
            yield env.process(env.faas.poll_available_replica(deployment.name))

        ps = []
        logging.info('executing requests')
        for deployment in self.deployments:
            fn_request_factory = self.fn_request_factories.get(deployment.name)
            if fn_request_factory is None:
                logger.info(f'No request factory for {deployment.name}')
                continue
            try:
                ia_generator = self.arrival_profiles[deployment.name]
                if self.duration is None:
                    p = env.process(
                        function_trigger(env, deployment, fn_request_factory, ia_generator, max_requests=1000))
                else:
                    p = env.process(function_trigger(env, deployment, fn_request_factory, ia_generator))
                ps.append(p)
            except KeyError:
                logging.warning('no arrival profile for deployment %s', deployment.name)

        if self.duration is not None:
            env.process(self.wait(env, ps))

        yield from ps

    def wait(self, env, ps):
        yield env.timeout(env.now + self.duration)
        for p in ps:
            if p.is_alive:
                p.interrupt('stop')


class DegradationBenchmarkBase(BenchmarkBase):

    def __init__(self, images: List[ImageProperties], deployments: List[SimFunctionDeployment],
                 arrival_profiles: Dict[str, Generator], fn_request_factories: Dict[str, FunctionRequestFactory],
                 duration: int = None, model_folder='./data'):
        super().__init__(images, deployments, arrival_profiles, fn_request_factories, duration)
        self.model_folder = model_folder

    def setup(self, env: Environment):
        super().setup(env)
        set_degradation(env, self.model_folder)


def get_model_file(folder, node_name):
    if 'xeongpu' in node_name or 'xeoncpu' in node_name:
        file = 'eb-xeongpu.sav'
    elif 'nx' in node_name:
        file = 'eb-jetson-nx-01.sav'
    elif 'nano' in node_name:
        file = 'eb-jetson-nano-01.sav'
    elif 'tx2' in node_name:
        file = 'eb-jetson-tx2-01.sav'
    elif 'tpu' in node_name or 'coral' in node_name:
        file = 'eb-rpi4-01.sav'
    elif 'rpi3' in node_name:
        file = 'eb-rpi3-01.sav'
    elif 'rockpi' in node_name:
        file = 'eb-rockpi.sav'
    elif 'nuc' in node_name:
        file = 'eb-nuc7.sav'
    elif 'rpi4' in node_name:
        file = 'eb-rpi4-01.sav'
    else:
        raise ValueError(f"Can't find model for node: {node_name}")
    return os.path.join(folder, file)


def set_degradation(env: Environment, folder: str):
    models = {}
    for ether_node in env.topology.get_nodes():
        try:
            name = ether_node.name[:ether_node.name.rindex("_")]
            model = models.get(name, None)
            if model is None:
                model_file = get_model_file(folder, name)
                model = loader.load_model(model_file)
                models[name] = model

            env.degradation_models[ether_node.name] = model
        except ValueError:
            # happens when an ether_node has no '_', i.e. docker registry -> can be ignored
            pass
