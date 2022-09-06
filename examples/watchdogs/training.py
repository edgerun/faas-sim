import logging
from typing import Generator, List

from faas.system.core import FunctionRequest

from sim import docker
from sim.core import Environment
from sim.faas import ForkingWatchdog, SimFunctionReplica
from sim.faas.watchdogs import WatchdogResponse

logger = logging.getLogger(__name__)


class TrainingFunctionSim(ForkingWatchdog):
    """
    Training forks per request
    Claims resources per request and downloads per request the model
    """

    def deploy(self, env: Environment, replica: SimFunctionReplica):
        # simulate a docker pull command for deploying the function (also done by sim.faassim.DockerDeploySimMixin)
        yield from docker.pull(env, replica.container.image, replica.node.ether_node)

    def startup(self, env: Environment, replica: SimFunctionReplica):
        logger.info('[simtime=%.2f] starting up function replica for function %s', env.now, replica.function.name)

        # you could create a very fine-grained setup routines here
        yield env.timeout(1)  # simulate docker startup

    def claim_resources(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest):
        cpu_index = env.resource_state.put_resource(replica, 'cpu', 0.7)
        memory_index = env.resource_state.put_resource(replica, 'memory', 0.3)
        yield env.timeout(0)
        return [cpu_index, memory_index]


    def release_resources(self, env: Environment, replica: SimFunctionReplica, resource_indices: List[int]):
        for idx in resource_indices:
            env.resource_state.remove_resource(replica, idx)
        yield env.timeout(0)

    def execute(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, WatchdogResponse]:
        # mock download, for actual network download simulation look at simulate_data_download
        yield env.timeout(1)

        # training
        yield env.timeout(5)

        # mock upload
        yield env.timeout(1)

        return WatchdogResponse(
            '',
            200,
            150
        )

    def teardown(self, env: Environment, replica: SimFunctionReplica):
        yield env.timeout(0)
