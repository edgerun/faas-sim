from faas.context import InMemoryDeploymentService
from faas.system import Metrics

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.context.platform.deployment.service import SimFunctionDeploymentService


def create_deployment_service(metrics: Metrics):
    in_memory_deployment_service = InMemoryDeploymentService[SimFunctionDeployment]([])
    return SimFunctionDeploymentService(in_memory_deployment_service, metrics)
