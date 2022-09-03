import logging
from typing import List

from ether.util import parse_size_string
from faas.system import DeploymentRanking, ScalingConfiguration
from faas.system import FunctionContainer, FunctionImage, Function
from faas.util.constant import controller_role_label, hostname_label, client_role_label, zone_label, function_label, \
    api_gateway_type_label, pod_type_label

from examples.decentralized_clients.deployments import prepare_resnet_inference_deployment, \
    prepare_resnet_training_deployment
from sim.context.platform.deployment.model import SimFunctionDeployment, SimScalingConfiguration
from sim.docker import ImageProperties
from sim.faas.core import SimResourceConfiguration, Node
from sim.faas.loadbalancers import LoadBalancerFunctionContainer, \
    ForwardingClientFunctionContainer
from sim.requestgen import expovariate_arrival_profile, constant_rps_profile

logger = logging.getLogger(__name__)

def create_load_balancer_deployment(lb_id: str, type: str, host: str, cluster: str):
    lb_fn_name = f'lb-' + lb_id + "-" + host
    lb_image_name = type
    lb_image = FunctionImage(image=lb_image_name)
    lb_fn = Function(lb_fn_name, fn_images=[lb_image],
                     labels={function_label: api_gateway_type_label, controller_role_label: 'true',
                             zone_label: cluster})

    fn_container = FunctionContainer(lb_image, SimResourceConfiguration(),
                                     {function_label: api_gateway_type_label, controller_role_label: 'true',
                                      hostname_label: host, zone_label: cluster,
                                      pod_type_label: api_gateway_type_label})

    lb_container = LoadBalancerFunctionContainer(fn_container)

    lb_fd = SimFunctionDeployment(
        lb_fn,
        [lb_container],
        SimScalingConfiguration(ScalingConfiguration(scale_min=1, scale_max=1, scale_zero=False)),
        DeploymentRanking([fn_container])
    )

    return lb_fd

def get_go_load_balancer_image_props(name: str) -> List[ImageProperties]:
    return [
        # image size from go-load-balancer
        ImageProperties(name, parse_size_string('10M'), arch='arm32v7'),
        ImageProperties(name, parse_size_string('10M'), arch='x86'),
        ImageProperties(name, parse_size_string('10M'), arch='arm64v8'),
    ]

def prepare_load_balancer_deployments(type: str, hosts: List[Node]) -> List[SimFunctionDeployment]:
    def create_id(i: int):
        return f'load-balancer-{i}'

    lbs = []
    for idx, host in enumerate(hosts):
        lb_id = create_id(idx)
        lbs.append(create_load_balancer_deployment(lb_id, type, host.name, host.labels[zone_label]))

    return lbs

def prepare_client_deployments_for_experiment(clients, lb_deployments: List[SimFunctionDeployment],
                                              deployments: List[SimFunctionDeployment]) -> List[
    SimFunctionDeployment]:

    fds = []
    inference_clients = [clients[0].name]
    training_clients = [clients[1].name]
    # we assume that the file size is 250KB
    inference_size = 250
    # we assume that the file size is 10MB (100000KB)
    training_size = 10000
    fds.extend(prepare_inference_clients(inference_clients, inference_size, deployments[0], lb_deployments[0]))
    fds.extend(prepare_training_clients(training_clients, training_size, deployments[1], lb_deployments[2]))

    return fds

def prepare_client_deployment(client_id: str, host: str, ia_generator, max_requests, size: int,
                              deployment: SimFunctionDeployment, lb_deployment: SimFunctionDeployment):
    # Design time
    client_fn_name = f'client-{deployment.name}-' + client_id + "-" + host
    client_image_name = 'galileo-worker'
    client_image = FunctionImage(image=client_image_name)
    client_fn = Function(client_fn_name, fn_images=[client_image])

    fn_container = FunctionContainer(client_image, SimResourceConfiguration(),
                                     {client_role_label: 'true',
                                      hostname_label: host})
    client_container = ForwardingClientFunctionContainer(
        fn_container,
        ia_generator,
        size,
        deployment,
        lb_deployment,
        max_requests=max_requests,
    )

    client_fd = SimFunctionDeployment(
        client_fn,
        [client_container],
        SimScalingConfiguration(ScalingConfiguration(scale_min=1, scale_max=1, scale_zero=False)),
        DeploymentRanking([client_container])
    )
    return client_fd

def prepare_client_deployments(ia_generator, client_names: List[str], deployment: SimFunctionDeployment,
                               lb_deployment: SimFunctionDeployment, max_requests: int, size: int):
    client_fds = []
    for idx, client in enumerate(client_names):
        client_inference_fd = prepare_client_deployment(
            str(idx),
            client,
            ia_generator,
            max_requests,
            size,
            deployment,
            lb_deployment
        )
        client_fds.append(client_inference_fd)

    return client_fds


def prepare_inference_clients(clients: List[str], size: int, deployment: SimFunctionDeployment,
                              lb_deployment: SimFunctionDeployment) -> List[
    SimFunctionDeployment]:
    inference_max_rps = 5
    inference_max_requests = 200
    # generate profile
    inference_ia_generator = expovariate_arrival_profile(constant_rps_profile(rps=inference_max_rps), max_ia=1)
    logger.info(
        f'executing resnet50-inference requests with {inference_max_rps} rps and maximum {inference_max_requests}')
    inference_client_fds = prepare_client_deployments(
        inference_ia_generator,
        clients,
        deployment,
        lb_deployment,
        inference_max_requests,
        size
    )
    return inference_client_fds


def prepare_training_clients(clients: List[str], size: int, deployment: SimFunctionDeployment,
                             lb_deployment: SimFunctionDeployment) -> List[
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
        deployment,
        lb_deployment,
        training_max_requests,
        size
    )

    return training_client_fds

def prepare_function_deployments() -> List[SimFunctionDeployment]:
    resnet_inference_fd = prepare_resnet_inference_deployment()

    resnet_training_fd = prepare_resnet_training_deployment()

    return [resnet_inference_fd, resnet_training_fd]
