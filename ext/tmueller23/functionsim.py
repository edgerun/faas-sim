import logging
from typing import Dict

import numpy as np

from sim.core import Environment
from sim.faas import FunctionSimulator, FunctionRequest, FunctionReplica, SimulatorFactory, \
    FunctionCharacterization, FunctionContainer, HTTPWatchdog
from sim.net import SafeFlow

logger = logging.getLogger(__name__)

client_node_label = 'client'
size_label = 'size'


class PowerPredictionSimulatorFactory(SimulatorFactory):

    def __init__(self, fn_characterizations: Dict[str, FunctionCharacterization]):
        self.fn_characterizations = fn_characterizations

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        workers = int(fn.labels['workers'])
        return PowerPredictionSimulator(workers, self.fn_characterizations[fn.image])


class PowerPredictionSimulator(HTTPWatchdog):

    def __init__(self, workers: int, characterization: FunctionCharacterization):
        super().__init__(workers)
        self.resources = {}
        self.characterization = characterization

    def setup(self, env: Environment, replica: FunctionReplica):
        super().setup(env, replica)
        yield env.timeout(0)

    def deploy(self, env: Environment, replica: FunctionReplica):
        # simulate a docker pull command for deploying the function (also done by sim.faassim.DockerDeploySimMixin)
        # yield from docker.pull(env, replica.container.image, replica.node.ether_node)
        yield env.timeout(0)

    def claim_resources(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        resource_characterization = self.characterization.resource_oracle.get_resources(replica.node.name,
                                                                                        replica.image)
        cpu = resource_characterization.cpu
        gpu = resource_characterization.gpu
        blkio = resource_characterization.blkio
        net = resource_characterization.net
        ram = resource_characterization.ram

        features = np.array([cpu, gpu, blkio, net, ram])
        reshaped = features.reshape(1, -1)
        # start_time = timeit.default_timer()
        power = env.power_models[replica.node.name].predict(reshaped)[0]
        # elapsed = (timeit.default_timer() - start_time)
        # metrics: Metrics = env.metrics
        # metrics.log('prediction_time', elapsed)

        self.resources[request.request_id] = {'cpu': cpu, 'gpu': gpu, 'blkio': blkio, 'net': net, 'ram': ram,
                                              'power': power}

        env.resource_state.put_resource(replica, 'cpu', cpu)
        env.resource_state.put_resource(replica, 'gpu', gpu)
        env.resource_state.put_resource(replica, 'blkio', blkio)
        env.resource_state.put_resource(replica, 'net', net)
        env.resource_state.put_resource(replica, 'ram', ram)
        env.resource_state.put_resource(replica, 'power', power)

        yield env.timeout(0)

    def release_resources(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        resource_characterization = self.resources[request.request_id]
        cpu = resource_characterization['cpu']
        gpu = resource_characterization['gpu']
        blkio = resource_characterization['blkio']
        net = resource_characterization['net']
        ram = resource_characterization['ram']
        power = resource_characterization['power']

        env.resource_state.remove_resource(replica, 'cpu', cpu)
        env.resource_state.remove_resource(replica, 'gpu', gpu)
        env.resource_state.remove_resource(replica, 'blkio', blkio)
        env.resource_state.remove_resource(replica, 'net', net)
        env.resource_state.remove_resource(replica, 'ram', ram)
        env.resource_state.remove_resource(replica, 'power', power)

        del self.resources[request.request_id]
        yield env.timeout(0)

    def execute(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        fet = self.characterization.sample_fet(replica.node.name)
        yield from self.data_download(env, replica)
        yield env.timeout(fet)

    def data_download(self, env: Environment, replica: FunctionReplica):
        client_node_name = replica.container.labels[client_node_label]
        client_node = env.cluster.get_node(client_node_name)
        node = replica.node
        route = env.topology.route_by_node_name(client_node.name, node.name)
        size = replica.container.labels[size_label]
        flow = SafeFlow(env, size, route)
        started = env.now
        yield flow.start()
        for hop in route.hops:
            env.metrics.log_network(size, 'data_download', hop)
        env.metrics.log_flow(size, env.now - started, route.source, route.destination, 'data_download')
