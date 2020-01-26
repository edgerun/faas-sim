import logging
from typing import List, Callable, NamedTuple

from core.clustercontext import ClusterContext
from sim.faas import FaasSimEnvironment, request_generator, FunctionRequest, empty, Function
from sim.simclustercontext import SimulationClusterContext
from sim.stats import ParameterizedDistribution
from sim.synth import pods
from sim.synth.bandwidth import generate_bandwidth_graph
from sim.synth.nodes import node_synthesizer, node_factory_cloud_majority, node_factory_50_percent_cloud


class Scenario:

    @property
    def until(self) -> int:
        raise NotImplementedError

    def cluster(self) -> ClusterContext:
        raise NotImplementedError

    def scenario_daemon(self, env: FaasSimEnvironment):
        yield env.timeout(0)


class FunctionBlueprint(NamedTuple):
    name: str
    pod_factory: Callable
    triggers: List[str] = list()

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 20
    scale_zero: bool = False


class TestScenario(Scenario):

    def __init__(self) -> None:
        super().__init__()
        logging.basicConfig(level=logging.DEBUG)

    def cluster(self) -> ClusterContext:
        gen = node_synthesizer(node_factory_cloud_majority)

        nodes = [next(gen) for i in range(100)]
        topology = generate_bandwidth_graph(nodes)
        cluster = SimulationClusterContext(nodes, topology)

        return cluster

    def scenario_daemon(self, env: FaasSimEnvironment):
        yield env.timeout(0)

        for function in env.functions.values():  # FIXME
            env.faas_gateway.deploy(function)

        yield env.timeout(10)

        training_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((300, 300,), None, None)),
            lambda: FunctionRequest('preprocess', empty)
        )
        inference_trigger = request_generator(
            env,
            ParameterizedDistribution.expon(((25, 50), None, None)),
            lambda: FunctionRequest('inference', empty)
        )

        env.process(training_trigger)
        env.process(inference_trigger)


class TestScenario2(Scenario):
    def __init__(self) -> None:
        super().__init__()
        # logging.basicConfig(level=logging.DEBUG)

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

        self.max_deployments = 50

    def cluster(self) -> ClusterContext:
        gen = node_synthesizer(node_factory_50_percent_cloud)

        nodes = [next(gen) for i in range(25)]
        topology = generate_bandwidth_graph(nodes)
        cluster = SimulationClusterContext(nodes, topology)

        return cluster

    def scenario_daemon(self, env: FaasSimEnvironment):
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
