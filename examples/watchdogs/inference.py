from typing import Generator, List

from faas.system.core import FunctionRequest

from sim import docker
from sim.core import Environment
from sim.faas import HTTPWatchdog, SimFunctionReplica, simulate_data_download, FunctionResourceCharacterization
from sim.faas.watchdogs import WatchdogResponse
from sim.oracle.oracle import ResourceOracle, FetOracle


class InferenceFunctionSim(HTTPWatchdog):
    """
    Inference model downloads and caches model at the beginning and  claims resources for HTTP server.
    During execution only inference happens.
    """

    def __init__(self, fet_oracle: FetOracle, resource_oracle: ResourceOracle):
        super().__init__(4)
        self.fet_oracle = fet_oracle
        self.resource_oracle = resource_oracle

    def deploy(self, env: Environment, replica: SimFunctionReplica):
        # simulate a docker pull command for deploying the function (also done by sim.faassim.DockerDeploySimMixin)
        yield from docker.pull(env, replica.container.image, replica.node.ether_node)

    def setup(self, env: Environment, replica: SimFunctionReplica):
        super().setup(env, replica)
        # the inference function mostly consumes memory by loading the model
        resources: FunctionResourceCharacterization = self.resource_oracle.get_resources(replica.node.name,replica.image)

        memory_resource_index = env.resource_state.put_resource(replica, 'memory', resources.ram)
        self.resource_indices = [memory_resource_index]

        yield from simulate_data_download(env, replica)

    def teardown(self, env: Environment, replica: SimFunctionReplica):
        yield from super(InferenceFunctionSim, self).teardown(env, replica)
        for resource_index in self.resource_indices:
            env.resource_state.remove_resource(replica, resource_index)
        yield env.timeout(0)

    def claim_resources(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest):
        resources: FunctionResourceCharacterization = self.resource_oracle.get_resources(replica.node.name, replica.image)
        # during invocation the inference function mostly consumes cpu
        cpu_resource_index = env.resource_state.put_resource(replica, 'cpu', resources.cpu)
        yield env.timeout(0)
        return [cpu_resource_index]

    def release_resources(self, env: Environment, replica: SimFunctionReplica, resource_indices: List[int]):
        for resource_index in resource_indices:
            env.resource_state.remove_resource(replica, resource_index)
        yield env.timeout(0)

    def execute(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, WatchdogResponse]:
        fet = self.fet_oracle.sample(replica.node.name, replica.image)
        yield env.timeout(fet)
        return WatchdogResponse(
            '',
            200,
            150
        )
