import logging
from typing import List

from ether.util import parse_size_string
from faas.system import FunctionContainer, FunctionImage, Function, \
    ScalingConfiguration
from faas.util.constant import client_role_label, hostname_label, worker_role_label, function_label, pod_type_label, \
    function_type_label

from examples.decentralized_clients.clients import ClientFunctionContainer
from sim.context.platform.deployment.model import SimFunctionDeployment, SimScalingConfiguration, DeploymentRanking
from sim.docker import ImageProperties
from sim.faas.core import SimResourceConfiguration
from sim.requestgen import expovariate_arrival_profile, constant_rps_profile

logger = logging.getLogger(__name__)


def prepare_client_deployment(client_id: str, host: str, ia_generator, max_requests, request_factory,
                              deployment: SimFunctionDeployment):
    # Design time
    client_fn_name = f'client-{deployment.name}-' + client_id + "-" + host
    client_image_name = 'galileo-worker'
    client_image = FunctionImage(image=client_image_name)
    client_fn = Function(client_fn_name, fn_images=[client_image])

    fn_container = FunctionContainer(client_image, SimResourceConfiguration(),
                                     {client_role_label: 'true',
                                      hostname_label: host})
    client_container = ClientFunctionContainer(fn_container, ia_generator, request_factory,
                                               deployment,
                                               max_requests=max_requests)

    client_fd = SimFunctionDeployment(
        client_fn,
        [client_container],
        SimScalingConfiguration(ScalingConfiguration(scale_min=1, scale_max=1, scale_zero=False)),
        DeploymentRanking([client_container])
    )
    return client_fd


def prepare_resnet_inference_deployment():
    # Design time

    resnet_inference = 'resnet50-inference'
    inference_cpu = 'resnet50-inference-cpu'

    resnet_inference_cpu = FunctionImage(image=inference_cpu)
    resnet_fn = Function(resnet_inference, fn_images=[resnet_inference_cpu], labels={function_label: resnet_inference})

    # Run time

    resnet_cpu_container = FunctionContainer(resnet_inference_cpu, SimResourceConfiguration(),
                                             {worker_role_label: "true", function_label: resnet_inference,
                                              pod_type_label: function_type_label})

    resnet_fd = SimFunctionDeployment(
        resnet_fn,
        [resnet_cpu_container],
        SimScalingConfiguration(),
        DeploymentRanking([resnet_cpu_container])
    )

    return resnet_fd


def prepare_resnet_training_deployment():
    # Design time

    resnet_training = 'resnet50-training'
    training_cpu = 'resnet50-training-cpu'

    resnet_training_cpu = FunctionImage(image=training_cpu)
    resnet_fn = Function(resnet_training, fn_images=[resnet_training_cpu], labels={function_label: resnet_training})

    # Run time

    resnet_cpu_container = FunctionContainer(resnet_training_cpu, SimResourceConfiguration(),
                                             labels={function_label: resnet_training, worker_role_label: 'true',
                                                     pod_type_label: function_type_label})

    resnet_fd = SimFunctionDeployment(
        resnet_fn,
        [resnet_cpu_container],
        SimScalingConfiguration(),
        DeploymentRanking([resnet_cpu_container])
    )

    return resnet_fd


def prepare_function_deployments() -> List[SimFunctionDeployment]:
    resnet_inference_fd = prepare_resnet_inference_deployment()

    resnet_training_fd = prepare_resnet_training_deployment()

    return [resnet_inference_fd, resnet_training_fd]


def get_resnet50_inference_cpu_image_properties():
    return [
        ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='arm32'),
        ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='x86'),
        ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='aarch64')
    ]


def get_galileo_worker_image_properties():
    return [
        ImageProperties('galileo-worker', parse_size_string('23M'), arch='arm32'),
        ImageProperties('galileo-worker', parse_size_string('23M'), arch='x86'),
        ImageProperties('galileo-worker', parse_size_string('23M'), arch='aarch64')
    ]


def get_resnet50_training_cpu_image_properties():
    return [
        ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='arm32'),
        ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='x86'),
        ImageProperties('resnet50-training-cpu', parse_size_string('128M'), arch='aarch64')
    ]


def prepare_client_deployments(ia_generator, client_names: List[str], deployment: SimFunctionDeployment,
                               max_requests: int, request_factory):
    client_fds = []
    for idx, client in enumerate(client_names):
        client_inference_fd = prepare_client_deployment(
            str(idx),
            client,
            ia_generator,
            max_requests,
            request_factory,
            deployment
        )
        client_fds.append(client_inference_fd)

    return client_fds


def prepare_inference_clients(clients: List[str], request_factory, deployment: SimFunctionDeployment) -> List[
    SimFunctionDeployment]:
    inference_max_rps = 5
    inference_max_requests = 200
    # generate profile
    inference_ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=inference_max_rps), max_ia=1)

    inference_client_fds = prepare_client_deployments(
        inference_ia_generator,
        clients,
        deployment, inference_max_requests,
        request_factory
    )
    return inference_client_fds


def prepare_training_clients(clients: List[str], request_factory, deployment: SimFunctionDeployment) -> List[
    SimFunctionDeployment]:
    training_max_rps = 1
    training_max_requests = 10
    logger.info(
        f'executing resnet50-training requests with {training_max_rps} rps and maximum {training_max_requests}')
    # generate profile
    training_ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=training_max_rps), max_ia=1)

    training_client_fds = prepare_client_deployments(
        training_ia_generator,
        clients,
        deployment, training_max_requests,
        request_factory
    )

    return training_client_fds
