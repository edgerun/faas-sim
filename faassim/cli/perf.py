import time

from skippy.core.priorities import BalancedResourcePriority, \
    LatencyAwareImageLocalityPriority, CapabilityPriority, DataLocalityPriority, LocalityTypePriority, \
    ImageLocalityPriority
from skippy.core.scheduler import Scheduler
from srds import BufferedSampler, IntegerSampler

from faassim.scenarios import CloudRegionsSynthesizer
from faassim.simclustercontext import SimulationClusterContext
from faassim.synth.pods import MLWorkflowPodSynthesizer

exp_01 = {
    'priorities': [
        (1, BalancedResourcePriority()),
    ],
    'percentage_of_nodes_to_score': 100
}

exp_02 = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
        (1, LatencyAwareImageLocalityPriority()),
        (1, LocalityTypePriority()),
        (1, DataLocalityPriority()),
    ]
}

exp_03 = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
        (1, LatencyAwareImageLocalityPriority()),
        (1, LocalityTypePriority()),
        (1, DataLocalityPriority()),
        (1, CapabilityPriority()),
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
        (1, LatencyAwareImageLocalityPriority()),
        (1, LocalityTypePriority()),
    ]
}

exp_04 = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
        (1, LatencyAwareImageLocalityPriority()),
        (1, LocalityTypePriority()),
        (1, DataLocalityPriority()),
        (1, CapabilityPriority()),
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
        (1, LatencyAwareImageLocalityPriority()),
        (1, LocalityTypePriority()),
        (1, DataLocalityPriority()),
        (1, CapabilityPriority()),
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
        (1, LatencyAwareImageLocalityPriority()),
    ]
}


def pod_gen(ctx: SimulationClusterContext):
    s = MLWorkflowPodSynthesizer(pareto=False)
    ctx.image_states = s.get_image_states()

    i = 0
    while True:
        i += 1
        yield from s.create_workflow_pods(i)


class RandomBwGraph:
    class RandomFloatDict:

        def __init__(self) -> None:
            self.sampler = BufferedSampler(IntegerSampler(10, 1000))

        def __getitem__(self, item):
            return self.sampler.sample()

    def __init__(self) -> None:
        self.rfd = self.RandomFloatDict()

    def __getitem__(self, item):
        return self.rfd


def main():
    print('num_constraints,num_nodes,sampling,duration')
    for perc in [0, 100]:
        for sched_params in [exp_01, exp_02, exp_03, exp_04]:
            sched_params['percentage_of_nodes_to_score'] = perc
            run_experiment(100, sched_params)
            run_experiment(500, sched_params)

            for i in range(0, 15):
                num_nodes = ((i + 1) * 1000)
                run_experiment(num_nodes, sched_params)


def run_experiment(num_nodes, sched_params):
    synth = CloudRegionsSynthesizer(regions=10, vms_per_region=int(num_nodes / 10))
    t = synth.create_topology()
    t._bandwidth_graph = RandomBwGraph()
    ctx = t.create_cluster_context()
    scheduler = Scheduler(ctx, **sched_params)
    run_loop(ctx, pod_gen(ctx), scheduler)


def run_loop(ctx, podgen, scheduler):
    num_constraints = len(scheduler.priorities)
    num_nodes = len(ctx.nodes)
    perc_score = scheduler.percentage_of_nodes_to_score
    for i in range(30):
        then = time.time()
        scheduler.schedule(next(podgen))
        now = time.time()
        print('%d,%d,%d,%.4f' % (num_constraints, num_nodes, perc_score, (now - then) * 1000))


if __name__ == '__main__':
    main()
