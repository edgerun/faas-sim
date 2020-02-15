import logging
import math
import os
import pickle
from abc import ABC
from typing import Tuple

import sim.synth.network as netsynth
import sim.synth.nodes as nodesynth
from core.model import Pod
from core.storage import DataItem
from core.utils import parse_size_string
from sim.faas import FaasSimEnvironment, request_generator, FunctionRequest, empty, Function, FunctionState
from sim.net import Topology, Link, Edge, Internet, Registry
from sim.stats import ParameterizedDistribution, PopulationSampler
from sim.synth.pods import MLWorkflowPodSynthesizer

logger = logging.getLogger(__name__)


class ClusterSynthesizer:
    internet = Internet
    registry = Registry

    def create_topology(self) -> Topology:
        raise NotImplementedError


class UrbanSensingClusterSynthesizer(ClusterSynthesizer):

    def __init__(self, cells=76, cloud_vms=None) -> None:
        super().__init__()
        self.cells = cells

        if cloud_vms is None:
            self.cloud_vms = math.ceil(self.cells * 21 * 0.03)  # 21 nodes per cell, 3% of nodes are VM nodes
        else:
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
            nodesynth.create_tegra_node(f'{zone}_1'),
            nodesynth.create_tegra_node(f'{zone}_2'),
            nodesynth.create_tegra_node(f'{zone}_3'),
            nodesynth.create_tegra_node(f'{zone}_4'),
            nodesynth.create_tegra_node(f'{zone}_5'),
            nodesynth.create_tegra_node(f'{zone}_6'),
            nodesynth.create_tegra_node(f'{zone}_7'),
            nodesynth.create_tegra_node(f'{zone}_8'),
            nodesynth.create_tegra_node(f'{zone}_9'),
            nodesynth.create_tegra_node(f'{zone}_10'),
        ]

        # FIXME: better up/downlink bandwidths
        edges_lan, uplink, downlink = netsynth.create_lan(cloudlet_nodes + aot_comm_pis,
                                                          downlink_bw=100, uplink_bw=25, internal_bw=1000, name=zone)
        uplink.tags['zone'] = 'edge'
        downlink.tags['zone'] = 'edge'

        nodes = cloudlet_nodes + aot_nodes
        nodesynth.set_zone(nodes, zone)
        return nodes, edges_pis + edges_lan, uplink, downlink


class IndustrialIoTSynthesizer(ClusterSynthesizer):

    def __init__(self, premises=10) -> None:
        super().__init__()
        self.premises = premises

    def create_topology(self) -> Topology:
        all_nodes = list()
        all_edges = list()
        t = Topology(all_nodes, all_edges)

        # create the internet
        all_nodes.append(self.internet)

        all_nodes.append(self.registry)
        all_edges.append(Edge(self.registry, self.internet, directed=True))

        for i in range(self.premises):
            nodes, edges = self.create_premises(i)
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        return t

    def create_premises(self, i):
        nodes = list()
        edges = list()

        name = f'edge_{i}'

        uplink = Link(250, tags={'type': 'uplink', 'name': name})
        downlink = Link(500, tags={'type': 'downlink', 'name': name})

        edges.append(Edge(uplink, self.internet, directed=True))
        edges.append(Edge(self.internet, downlink, directed=True))

        switch = f'switch_{name}'
        edges.append(Edge(downlink, switch, directed=True))
        edges.append(Edge(switch, uplink, directed=True))

        # sbc edge cell
        nodes_1, edges_1 = self._create_sbc_wifi(i, switch)
        nodes.extend(nodes_1)
        edges.extend(edges_1)

        nodes_2, edges_2 = self._create_cell(i, switch)
        nodes.extend(nodes_2)
        edges.extend(edges_2)

        nodes_3, edges_3 = self._create_cloud(i, switch, 4)
        nodes.extend(nodes_3)
        edges.extend(edges_3)

        return nodes, edges

    def _create_cloud(self, i, switch, n=4):
        nodes = list()

        for j in range(n):
            nodes.append(nodesynth.create_cloud_node(prefix=f'edge_{i}_{j}'))

        nodes.append(nodesynth.mark_storage_node(nodesynth.create_cloud_node(prefix=f'edge_{i}_storage')))

        edges, egress, ingress = netsynth.create_lan(nodes, 1000, 1000, internal_bw=10000, name=f'cloud_{i}')

        nodesynth.set_zone(nodes, 'cloud')
        egress.tags['zone'] = 'cloud'
        ingress.tags['zone'] = 'cloud'

        edges.append(Edge(egress, switch, directed=True))
        edges.append(Edge(switch, ingress, directed=True))

        return nodes, edges

    def _create_cell(self, i, switch):
        name = f'edge_{i}'
        nodes = list()
        edges = list()

        nodes.append(nodesynth.mark_storage_node(nodesynth.create_nuc_node(name + '_storage')))
        nodes.append(nodesynth.create_nuc_node(name))
        nodes.append(nodesynth.create_tegra_node(name))

        for node in nodes:
            nodelink = Link(10000, tags={'name': f'{node}', 'type': 'node'})
            edges.append(Edge(node, nodelink))
            edges.append(Edge(nodelink, switch))

        return nodes, edges

    def _create_sbc_wifi(self, i, switch):
        name = f'wifi_edge_{i}'
        nodes = list()
        edges = list()

        # create wifi link, and a link from the AP to the switch
        wifi = Link(bandwidth=300, tags={'type': 'wifi', 'name': f'{name}'})
        ap_link = Link(bandwidth=1000, tags={'type': 'lan', 'name': f'{name}_ap'})

        edges.append(Edge(wifi, ap_link))
        edges.append(Edge(ap_link, switch))

        for j in range(4):
            node = nodesynth.create_rpi3_node(prefix=f'edge_{i}_{j}')
            nodes.append(node)
            # connect each node to the wifi
            edges.append(Edge(node, wifi))

        return nodes, edges


class CloudRegionsSynthesizer(ClusterSynthesizer):
    def __init__(self, regions=3, vms_per_region=150):
        self.regions = regions
        self.vms_per_region = vms_per_region

    def create_topology(self) -> Topology:
        all_nodes = list()
        all_edges = list()
        t = Topology(all_nodes, all_edges)

        # create the internet
        all_nodes.append(self.internet)

        # create the registry and attach to each cloud's switch (simulates a CDN, where a registry mirror resides in
        # each cloud region)
        all_nodes.append(self.registry)

        for r in range(self.regions):
            if r == 0:
                perc = 0.5
            elif r == 1:
                perc = 0.25
            elif r == 2:
                perc = 0.25
            else:
                raise ValueError

            nodes, edges = self._create_cloud(r, round(self.vms_per_region * perc))
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        return t

    def _create_cloud(self, r, n):
        name = f'cloud_{r}'

        nodes = list()
        for i in range(n):
            node = nodesynth.create_cloud_node(f'{name}_{i}')
            nodes.append(node)

        if r == 0:
            perc_storage = 0.10
        else:
            perc_storage = 0.05

        num_storage = math.ceil(n * perc_storage)

        for i in range(num_storage):
            nodesynth.mark_storage_node(nodes[i])

        # 10 GBit/s lan and a 1Gbit up/down
        # https://medium.com/slalom-technology/examining-cross-region-communication-speeds-in-aws-9a0bee31984f
        edges, uplink, downlink = netsynth.create_lan(nodes, 1000, 1000, internal_bw=10000, name=name)

        nodesynth.set_zone(nodes, 'cloud')
        uplink.tags['zone'] = 'cloud'
        downlink.tags['zone'] = 'cloud'

        edges.append(Edge(uplink, self.internet, directed=True))
        edges.append(Edge(self.internet, downlink, directed=True))

        # link the registry directly to the datacenter switch
        switch = f'switch_{name}'  # comes from nodesynth.create_lan
        edges.append(Edge(self.registry, switch, True))

        return nodes, edges


class Scenario(ABC):

    def topology(self) -> Topology:
        raise NotImplementedError

    def scenario_daemon(self, env: FaasSimEnvironment):
        yield env.timeout(0)

    def scheduler_parameters(self) -> dict:
        return {}

    @classmethod
    def lazy(cls, *args, **kwargs):
        file = '/tmp/schedsim-scenario--' + cls.__name__ + '.pkl'
        try:
            with open(file, 'rb') as fd:
                scenario = pickle.load(fd)
                return scenario
        except FileNotFoundError:
            pass

        scenario = cls(*args, **kwargs)
        scenario.topology()

        with open(file, 'wb') as fd:
            pickle.dump(scenario, fd)

        return scenario

    @classmethod
    def purge(cls):
        file = '/tmp/schedsim-scenario--' + cls.__name__ + '.pkl'
        os.remove(file)


class EvaluationScenario(Scenario, ABC):
    def __init__(self, max_deployments, max_invocations) -> None:
        super().__init__()
        self.max_deployments = max_deployments
        self.max_invocations = max_invocations
        self._topology = None
        self._pod_synthesizer = None

    def scenario_daemon(self, env: FaasSimEnvironment):
        if self._pod_synthesizer is None:
            self._pod_synthesizer = MLWorkflowPodSynthesizer(max_image_variety=self.max_deployments, pareto=True)

        # first, create all deployments, which will synthesize images and allow us to get the image states
        deployments = [(i,) + self._pod_synthesizer.create_workflow_pods(i) for i in range(self.max_deployments)]
        env.cluster.image_states = self._pod_synthesizer.get_image_states()

        self.distribute_buckets(env, deployments)
        self.initialize_data(env, deployments)

        yield env.timeout(0)

        for deployment in deployments:
            yield from self.inject_deployment(env, deployment, blocking=False, interval=250)
            yield env.timeout(5)
            self.inject_workload_generator(env, deployment)

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

    def inject_deployment(self, env, deployment: Tuple[int, Pod, Pod, Pod], blocking=True, interval=0):
        i, pod0, pod1, pod2 = deployment

        logger.debug('%.2f injecting deployment %d (blocking=%s)', env.now, i, blocking)

        fn0_name = f'wf_0_preprocess_{i}'
        fn1_name = f'wf_1_train_{i}'
        fn2_name = f'wf_2_inference_{i}'

        fn0 = Function(name=fn0_name, pod=pod0, triggers=[fn1_name])
        fn0.scale_max = 1
        fn0.scale_zero = True

        fn1 = Function(name=fn1_name, pod=pod1)
        fn1.scale_max = 1
        fn1.scale_zero = True

        fn2 = Function(name=fn2_name, pod=pod2)

        # deploy functions
        for fn in (fn0, fn1, fn2):
            env.faas_gateway.deploy(fn)

        # optionally wait
        if blocking:
            for fn in (fn0, fn1, fn2):
                while fn.state == FunctionState.STARTING or fn.state == FunctionState.CONCEIVED:
                    yield env.timeout(1)

        yield env.timeout(interval)

    def inject_workload_generator(self, env, deployment: Tuple[int, Pod, Pod, Pod]):
        i, pod0, pod1, pod2 = deployment

        logger.debug('%.2f injecting request generators for deployment %d', env.now, i)

        prep_function_name = f'wf_0_preprocess_{i}'
        training_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((200, 100,), None, None)),
            lambda: FunctionRequest(prep_function_name, empty)
        )

        inference_function_name = f'wf_2_inference_{i}'
        inference_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((25, 50,), None, None)),
            lambda: FunctionRequest(inference_function_name, empty)
        )

        env.process(training_trigger)
        env.process(inference_trigger)

    def initialize_data(self, env: FaasSimEnvironment, deployments):
        for i, _, _, _ in deployments:
            bucket = f'bucket_{i}'

            raw_data = DataItem(bucket, 'raw_data', parse_size_string('12Mi'))
            train_data = DataItem(bucket, 'train_data', parse_size_string('209Mi'))
            model = DataItem(bucket, 'model', parse_size_string('1500Ki'))

            env.cluster.storage_index.put(raw_data)
            env.cluster.storage_index.put(train_data)
            env.cluster.storage_index.put(model)

    def distribute_buckets(self, env: FaasSimEnvironment, deployments):
        sampler = PopulationSampler(list(env.cluster.storage_nodes.keys()))

        # each deployment gets randomly (uniformly) assigned a storage node
        storage_nodes = sampler.sample(len(deployments))
        assignment = zip(storage_nodes, deployments)

        for node, deployment in assignment:
            i = deployment[0]
            bucket_name = f'bucket_{i}'
            # create a bucket belonging to the workflow on the randomly assigned storage node
            env.cluster.storage_index.mb(bucket_name, node)


class UrbanSensingScenario(EvaluationScenario):

    def __init__(self, cells=1, deployments=None, max_invocations=None) -> None:
        self.cells = cells
        deployments = deployments or cells * 10
        max_invocations = max_invocations or (deployments ** 1.1) * 400
        super().__init__(deployments, max_invocations)

    def topology(self) -> Topology:
        if self._topology:
            return self._topology

        synth = UrbanSensingClusterSynthesizer(cells=self.cells)
        self._topology = synth.create_topology()
        self._topology.create_index()
        self._topology.get_bandwidth_graph()

        return self._topology


class IndustrialIoTScenario(EvaluationScenario):

    def __init__(self, premises=10, deployments=None, max_invocations=None) -> None:
        self.premises = premises
        deployments = deployments or premises * 5
        max_invocations = max_invocations or (deployments ** 1.1) * 400
        super().__init__(deployments, max_invocations)

    def topology(self) -> Topology:
        if self._topology:
            return self._topology

        synth = IndustrialIoTSynthesizer(premises=self.premises)
        self._topology = synth.create_topology()
        self._topology.create_index()
        self._topology.get_bandwidth_graph()

        return self._topology


class CloudRegionScenario(EvaluationScenario):

    def __init__(self, vms_per_region=150, deployments=None, max_invocations=None) -> None:
        self.vms_per_region = vms_per_region
        deployments = deployments or (int(vms_per_region * 3 * 0.5))
        max_invocations = max_invocations or (deployments ** 1.3) * 400
        super().__init__(deployments, max_invocations)

    def topology(self) -> Topology:
        if self._topology:
            return self._topology

        synth = CloudRegionsSynthesizer(vms_per_region=self.vms_per_region)
        self._topology = synth.create_topology()
        self._topology.create_index()
        self._topology.get_bandwidth_graph()

        return self._topology
