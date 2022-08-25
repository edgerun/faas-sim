from typing import List, Optional, Dict

from faas.context import InMemoryDeploymentService, FunctionDeploymentService
from faas.system import Metrics, FunctionContainer

from sim.context.platform.deployment.model import SimFunctionDeployment


class SimFunctionDeploymentService(FunctionDeploymentService[SimFunctionDeployment]):

    def __init__(self, in_memory_deployment_service: InMemoryDeploymentService[SimFunctionDeployment],
                 metrics: Metrics):
        self.metrics = metrics
        self.in_memory_deployment_service: InMemoryDeploymentService[
            SimFunctionDeployment] = in_memory_deployment_service

    def get_deployments(self) -> List[SimFunctionDeployment]:
        return self.in_memory_deployment_service.get_deployments()

    def get_by_name(self, fn_name: str) -> Optional[SimFunctionDeployment]:
        return self.in_memory_deployment_service.get_by_name(fn_name)

    def exists(self, name: str) -> bool:
        return self.in_memory_deployment_service.exists(name)

    def add(self, deployment: SimFunctionDeployment):
        # self.metrics.log_function_deployment(deployment)
        self.in_memory_deployment_service.add(deployment)

    def remove(self, function_name: str):
        deployment = self.get_by_name(function_name)
        # self.metrics.log_function_deployment_remove(deployment)
        self.in_memory_deployment_service.remove(function_name)

    def get_function_containers_by_name(self) -> Dict[str, FunctionContainer]:
        deployments: List[SimFunctionDeployment] = self.get_deployments()
        function_containers = {}
        for deployment in deployments:
            for fn_container in deployment.fn_containers:
                function_containers[fn_container.image] = fn_container
        return function_containers
