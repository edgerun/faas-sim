import logging
import pickle
from abc import ABC
from typing import List, Callable, NamedTuple

import sim.synth.network as netsynth
import sim.synth.nodes as nodesynth
from sim.faas import FaasSimEnvironment, request_generator, FunctionRequest, empty, Function, FunctionState
from sim.net import Topology, Link, Edge, Internet, Registry
from sim.stats import ParameterizedDistribution
from sim.synth import pods

logger = logging.getLogger(__name__)


class ClusterSynthesizer:

    def create_topology(self) -> Topology:
        raise NotImplementedError


class UrbanSensingClusterSynthesizer(ClusterSynthesizer):
    internet = Internet

    def __init__(self, cells=76, cloud_vms=30) -> None:
        super().__init__()
        self.cells = cells
        self.cloud_vms = cloud_vms

    def create_topology(self) -> Topology:
        all_nodes = list()
        all_edges = list()
        t = Topology(all_nodes, all_edges)

        # create the internet
        internet = self.internet
        all_nodes.append(internet)

        # create the registry and attach to the internet with essentially infinite bandwidth
        registry = Registry
        all_nodes.append(registry)
        # registry_link = Link(10 ** 12, tags={'name': 'registry', 'type': 'registry'})
        all_edges.append(Edge(registry, internet))

        # create cells
        for i in range(self.cells):
            nodes, edges, uplink, downlink = self._create_cell(i)

            all_nodes.extend(nodes)
            all_edges.extend(edges)
            all_edges.append(Edge(uplink, internet, directed=True))
            all_edges.append(Edge(internet, downlink, directed=True))

        # create cloud
        # nodes, edges = self._create_sparse_cloud(self.cloud_vms)
        nodes, edges = self._create_cloud(self.cloud_vms)

        all_nodes.extend(nodes)
        all_edges.extend(edges)

        return t

    def _create_sparse_cloud(self, n):
        switch = 'switch_cloud'

        nodes = list()
        edges = list()

        for i in range(n):
            nodes.append(nodesynth.create_cloud_node(i))

        for node in nodes:
            ingress = Link(bandwidth=10000, tags={'type': 'uplink', 'zone': 'cloud', 'name': node.name})
            egress = Link(bandwidth=10000, tags={'type': 'downlink', 'zone': 'cloud', 'name': node.name})

            edges.append(Edge(node, ingress))
            edges.append(Edge(node, egress))
            edges.append(Edge(switch, ingress, directed=True))
            edges.append(Edge(egress, switch, directed=True))

        edges.append(Edge(switch, self.internet))

        return nodes, edges

    def _create_cloud(self, n):
        nodes = list()
        for i in range(n):
            nodes.append(nodesynth.create_cloud_node(i))

        # 10 GBit/s lan and a lot of up/down
        # FIXME: upgrade to 10Gbit/s
        edges, uplink, downlink = netsynth.create_lan(nodes, 1e10, 1e10, internal_bw=10000, name='cloud')

        nodesynth.set_zone(nodes, 'cloud')
        uplink.tags['zone'] = 'cloud'
        downlink.tags['zone'] = 'cloud'

        edges.append(Edge(uplink, self.internet, directed=True))
        edges.append(Edge(self.internet, downlink, directed=True))

        return nodes, edges

    def _create_cell(self, i, num_aot_nodes=5):
        # TODO: parameterize makeup
        zone = f'edge_{i}'

        # https://arrayofthings.github.io/node.htmls
        aot_nodes = []  # all pis in a node (connected via lan)
        aot_comm_pis = []  # pis for communication (connected via wifi to the uplink)

        edges_pis = list()
        for a in range(num_aot_nodes):
            pi1 = nodesynth.create_rpi3_node(f'{zone}_{a}_sensors')
            pi2 = nodesynth.create_rpi3_node(f'{zone}_{a}_comm')
            aot_nodes.append(pi1)
            aot_nodes.append(pi2)
            aot_comm_pis.append(pi2)
            # connect the pis via a p2p network
            link = Link(1000, tags={'name': f'{pi1.name}_internal'})
            edges_pis.append(Edge(pi1, link))
            edges_pis.append(Edge(pi2, link))

        cloudlet_nodes = [
            nodesynth.mark_storage_node(nodesynth.create_nuc_node(f'{zone}_storage')),
            nodesynth.create_nuc_node(f'{zone}_1'),
            nodesynth.create_nuc_node(f'{zone}_2'),
            nodesynth.create_tegra_node(f'{zone}_1'),
            nodesynth.create_tegra_node(f'{zone}_2')
        ]

        # FIXME: better up/downlink bandwidths
        edges_lan, uplink, downlink = netsynth.create_lan(cloudlet_nodes + aot_comm_pis,
                                                          downlink_bw=100, uplink_bw=25, internal_bw=1000, name=zone)
        uplink.tags['zone'] = 'edge'
        downlink.tags['zone'] = 'edge'

        nodes = cloudlet_nodes + aot_nodes
        nodesynth.set_zone(nodes, zone)
        return nodes, edges_pis + edges_lan, uplink, downlink


class Scenario(ABC):

    def topology(self) -> Topology:
        raise NotImplementedError

    def scenario_daemon(self, env: FaasSimEnvironment):
        yield env.timeout(0)

    def scheduler_parameters(self) -> dict:
        return {}

    @classmethod
    def lazy(cls):
        file = '/tmp/schedsim-scenario--' + cls.__name__ + '.pkl'
        try:
            with open(file, 'rb') as fd:
                scenario = pickle.load(fd)
                return scenario
        except FileNotFoundError:
            pass

        scenario = cls()
        scenario.topology()

        with open(file, 'wb') as fd:
            pickle.dump(scenario, fd)

        return scenario


class FunctionBlueprint(NamedTuple):
    name: str
    pod_factory: Callable
    triggers: List[str] = list()

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 20
    scale_zero: bool = False


class TestScenario2(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self.max_deployments = 100
        self._topology = None

    def topology(self) -> Topology:
        if self._topology:
            return self._topology

        synth = UrbanSensingClusterSynthesizer(cells=10, cloud_vms=5)  # FIXME
        self._topology = synth.create_topology()
        self._topology.create_index()
        self._topology.get_bandwidth_graph()

        return self._topology

    def scenario_daemon(self, env: FaasSimEnvironment):
        self.blueprint_prep = FunctionBlueprint(
            'wf_0_preprocess_{i}', pods.create_ml_wf_1_pod, ['wf_1_train_{i}'],
            scale_max=1, scale_zero=True
        )
        self.blueprint_train = FunctionBlueprint(
            'wf_1_train_{i}', pods.create_ml_wf_2_pod,
            scale_max=1, scale_zero=True
        )
        self.blueprint_inference = FunctionBlueprint(
            'wf_2_inference_{i}', pods.create_ml_wf_3_serve
        )

        yield env.timeout(0)

        def deployment_injector():
            for i in range(self.max_deployments):
                self.inject_deployment(env, i)
                yield env.timeout(100)

        env.process(deployment_injector())

    def inject_deployment(self, env, i):
        cnt = 1
        for blueprint in [self.blueprint_prep, self.blueprint_train, self.blueprint_inference]:
            pod = i * 3 + cnt

            fn = Function(
                blueprint.name.format(i=i),
                blueprint.pod_factory(pod),
                [trigger.format(i=i) for trigger in blueprint.triggers],
            )

            if blueprint.scale_min:
                fn.scale_min = blueprint.scale_min
            if blueprint.scale_max:
                fn.scale_max = blueprint.scale_max
            if blueprint.scale_factor:
                fn.scale_factor = blueprint.scale_factor
            if blueprint.scale_zero:
                fn.scale_zero = blueprint.scale_zero

            cnt += 1
            env.faas_gateway.deploy(fn)

        prep_function_name = self.blueprint_prep.name.format(i=i)
        training_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((200, 100,), None, None)),
            lambda: FunctionRequest(prep_function_name, empty)
        )

        inference_function_name = self.blueprint_inference.name.format(i=i)
        inference_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((25, 50,), None, None)),
            lambda: FunctionRequest(inference_function_name, empty)
        )

        env.process(training_trigger)
        env.process(inference_trigger)


class EvaluationScenario(Scenario, ABC):
    def __init__(self, max_deployments, max_invocations) -> None:
        super().__init__()
        self.max_deployments = max_deployments
        self.max_invocations= max_invocations
        self._topology = None

    def scenario_daemon(self, env: FaasSimEnvironment):
        self.blueprint_prep = FunctionBlueprint(
            'wf_0_preprocess_{i}', pods.create_ml_wf_1_pod, ['wf_1_train_{i}'],
            scale_max=1, scale_zero=True
        )
        self.blueprint_train = FunctionBlueprint(
            'wf_1_train_{i}', pods.create_ml_wf_2_pod,
            scale_max=1, scale_zero=True
        )
        self.blueprint_inference = FunctionBlueprint(
            'wf_2_inference_{i}', pods.create_ml_wf_3_serve
        )

        yield env.timeout(0)

        for i in range(self.max_deployments):
            logger.debug('%.2f creating deployment %d', env.now, i)
            yield from self.inject_deployment(env, i, blocking=False)
            yield env.timeout(1)
            self.inject_workload_generator(env, i)

        logger.info('%.2f waiting for %d invocation', env.now, self.max_invocations)
        while env.metrics.total_invocations < self.max_invocations:
            logger.debug('%.2f %d invocations left', env.now, self.max_invocations - env.metrics.total_invocations)
            yield env.timeout(10)

        logger.info('%.2f done', env.now)

        # def deployment_injector():
        #     for i in range(self.max_deployments):
        #         yield from self.inject_deployment(env, i)
        #
        # def workload():
        #     for i in range(self.max_deployments):
        #         self.inject_workload_generator(env, i)
        #         yield env.timeout(10)
        #
        #     logger.info('%.2f running workload for %d', env.now, self.until)
        #     yield env.timeout(self.until)  # create workload
        #     logger.info('%.2f workload stopped', env.now)
        #
        # logger.info('%.2f starting deployment injector', env.now)
        # yield env.process(deployment_injector())
        # logger.info('%.2f starting workload injector', env.now)
        # yield env.process(workload())
        # logger.info('%.2f finished!', env.now)

    def inject_deployment(self, env, i, blocking=True):
        logger.debug('%.2f injecting deployment %s', env.now, i)

        cnt = 1
        for blueprint in [self.blueprint_prep, self.blueprint_train, self.blueprint_inference]:
            pod = i * 3 + cnt

            fn = Function(
                blueprint.name.format(i=i),
                blueprint.pod_factory(pod),
                [trigger.format(i=i) for trigger in blueprint.triggers],
            )

            if blueprint.scale_min:
                fn.scale_min = blueprint.scale_min
            if blueprint.scale_max:
                fn.scale_max = blueprint.scale_max
            if blueprint.scale_factor:
                fn.scale_factor = blueprint.scale_factor
            if blueprint.scale_zero:
                fn.scale_zero = blueprint.scale_zero

            cnt += 1
            env.faas_gateway.deploy(fn)

            if blocking:
                while fn.state == FunctionState.STARTING or fn.state == FunctionState.CONCEIVED:
                    yield env.timeout(1)
            else:
                yield env.timeout(0)

    def inject_workload_generator(self, env, i):
        logger.debug("%.2f injecting request generators for %d", env.now, i)
        prep_function_name = self.blueprint_prep.name.format(i=i)
        training_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((200, 100,), None, None)),
            lambda: FunctionRequest(prep_function_name, empty)
        )

        inference_function_name = self.blueprint_inference.name.format(i=i)
        inference_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((25, 50,), None, None)),
            lambda: FunctionRequest(inference_function_name, empty)
        )

        env.process(training_trigger)
        env.process(inference_trigger)


class UrbanSensingScenario(EvaluationScenario):

    def __init__(self) -> None:
        super().__init__(150, 20000)

    def topology(self) -> Topology:
        if self._topology:
            return self._topology

        synth = UrbanSensingClusterSynthesizer(cells=5, cloud_vms=2)  # FIXME
        self._topology = synth.create_topology()
        self._topology.create_index()
        self._topology.get_bandwidth_graph()

        return self._topology
