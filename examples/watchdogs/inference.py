from typing import Generator, List

from faas.system.core import FunctionRequest

from sim import docker
from sim.core import Environment
from sim.faas import HTTPWatchdog, SimFunctionReplica, simulate_data_download
from sim.faas.watchdogs import WatchdogResponse


class InferenceFunctionSim(HTTPWatchdog):
    """
    Inference model downloads and caches model at the beginning and  claims resources for HTTP server.
    During execution only inference happens.
    """

    def deploy(self, env: Environment, replica: SimFunctionReplica):
        # simulate a docker pull command for deploying the function (also done by sim.faassim.DockerDeploySimMixin)
        yield from docker.pull(env, replica.container.image, replica.node.ether_node)

    def setup(self, env: Environment, replica: SimFunctionReplica):
        super().setup(env, replica)
        # basic cpu usage, in %
        cpu_resource_index = env.resource_state.put_resource(replica, 'cpu', 0.08)

        # basic memory consumption, in MB
        memory_resource_index = env.resource_state.put_resource(replica, 'memory', 200)

        self.resource_indices = [cpu_resource_index, memory_resource_index]

        yield from simulate_data_download(env, replica)

    def teardown(self, env: Environment, replica: SimFunctionReplica):
        yield from super(InferenceFunctionSim, self).teardown(env, replica)
        for resource_index in self.resource_indices:
            env.resource_state.remove_resource(replica, resource_index)
        yield env.timeout(0)

    def claim_resources(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest):
        # no setup time, no memory because everything is cached - only cpu usage
        cpu_resource_index = env.resource_state.put_resource(replica, 'cpu', 0.2)
        memory_resource_index = env.resource_state.put_resource(replica, 'memory', 50)
        yield env.timeout(0)
        return [cpu_resource_index, memory_resource_index]

    def release_resources(self, env: Environment, replica: SimFunctionReplica, resource_indices: List[int]):
        for resource_index in resource_indices:
            env.resource_state.remove_resource(replica, resource_index)
        yield env.timeout(0)

    def execute(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, WatchdogResponse]:
        yield env.timeout(0.2)
        return WatchdogResponse(
            '',
            200,
            150
        )
