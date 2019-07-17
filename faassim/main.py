import argparse
import itertools
import time
from typing import NamedTuple, Dict, List, Generator, Tuple
import simpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from enum import Enum

from core.clustercontext import ClusterContext
from core.priorities import Priority, BalancedResourcePriority, ImageLocalityPriority
from core.scheduler import Scheduler
from sim.oracle import ExecutionTimeOracle, PlacementTimeOracle, Oracle
from sim.simclustercontext import SimulationClusterContext
from sim.stats import exp_sampler
from sim.synth_bandwidth import generate_bandwidth_graph
from sim.synth_nodes import node_synthesizer
from sim.synth_pods import pod_synthesizer, PodSynthesizer


class EventType(Enum):
    POD_QUEUED = "pod_queued",
    POD_RECEIVED = "pod_received",
    POD_SCHEDULED = "pod_scheduled"

    # The order is actually also alphabetically, so just implement lt as < on the name
    def __lt__(self, other):
        return self.name < other.name


class LoggingRow(NamedTuple):
    timestamp: float                 # When did it happen?
    event: EventType                 # What happened?
    value: str                       # Which pod was affected?
    additional_attributes: Dict = {} # What else could be interesting?


def run_load_generator(env: simpy.Environment, queue: simpy.Store, pod_synth: PodSynthesizer,
                       ia_sampler: Generator[float, float, None], log: List[LoggingRow]):
    """
    :param env: simpy environment
    :param queue: the work queue
    :param pod_synth: fake Pod generator
    :param ia_sampler: arrival profile
    :param log: simple array to append log messages
    :return:
    """
    while True:
        ia = next(ia_sampler)  # inter-arrival
        ia = round(ia, 3)  # millisecond accuracy
        yield env.timeout(ia)

        pod = next(pod_synth)
        queue.put(pod)

        logging.debug('pod arrived at %.2f seconds' % env.now)
        log.append(LoggingRow(env.now, EventType.POD_QUEUED, pod.name, {'queue_length': len(queue.items)}))


def run_scheduler_worker(env: simpy.Environment, queue: simpy.Store, context: ClusterContext, scheduler: Scheduler,
                         oracles: List[Oracle], log: List[LoggingRow]):
    while True:
        logging.debug('Scheduler waiting for pod...')
        pod = yield queue.get()

        # TODO fix time not changing (env.now)
        logging.debug('Pod received by scheduler at %.2f', env.now)
        log.append(LoggingRow(env.now, EventType.POD_RECEIVED, pod.name))
        then = time.time()

        # execute scheduling algorithm
        result = scheduler.schedule(pod)

        duration = ((time.time() - then) * 1000)
        yield env.timeout(duration / 1000)
        logging.debug('Pod scheduling took %.2f ms, and yielded %s', duration, result)

        # weight the placement
        metadata = dict([o.estimate(context, pod, result.suggested_host) for o in oracles])

        log.append(LoggingRow(env.now, EventType.POD_SCHEDULED, pod.name, metadata))


def simulate(cluster_context: ClusterContext, scheduler: Scheduler) -> pd.DataFrame:
    log = []
    oracles = [PlacementTimeOracle(), ExecutionTimeOracle()]
    env = simpy.RealtimeEnvironment(factor=0.01, strict=False)
    queue = simpy.Store(env)
    env.process(run_load_generator(env, queue, pod_synthesizer(), exp_sampler(lambd=1.5), log))
    env.process(run_scheduler_worker(env, queue, cluster_context, scheduler, oracles, log))
    env.sync()
    env.run(until=200)
    data = pd.DataFrame(data=log)
    return data


def plot_placement_time_cdf(df_1: pd.DataFrame, scheduler_name: str):
    # Only take the POD_QUEUED and the pod_scheduled events
    events = df_1['event']
    filtered = events.isin([EventType.POD_RECEIVED, EventType.POD_SCHEDULED])
    df_1 = df_1.loc[filtered]
    # Filter pod events of pods which have not fully been scheduled (not all 3 events are included)
    df_1 = df_1.groupby(['value']).filter(lambda x: len(x) == 2)
    # Convert the podname to an int (to allow proper sorting)
    df_1['value'] = df_1['value'].str[4:].astype(int)
    # Sort by pods, then by event (POD_QUEUED < POD_SCHEDULED)
    df_1 = df_1.sort_values(['value', 'event'], ascending=[True, True])
    # Drop the diff between two pods (every second entry)
    ser = df_1['timestamp'].diff().iloc[1::2]
    # Adopt the index
    ser.index = range(len(ser))
    # Transform from seconds to milliseconds
    ser = ser * 1000

    # Create the CDF of the series and plot it
    x, y = sorted(ser), np.arange(len(ser)) / len(ser)
    plt.plot(x, y, label=f'placement time using the {scheduler_name} scheduler')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Task Placement Latency (ms)')
    plt.savefig(f'results/sim_{scheduler_name}_placement_time_cdf.png')
    plt.show()


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Skippy Simulator')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug',
                        help='Enable debug logs.', default=False)
    parser.add_argument('-s', '--simulate', action='store_true', dest='simulate',
                        help='Only simulate the scheduling.', default=False)
    parser.add_argument('-p', '--plot', action='store_true', dest='plot',
                        help='Only plot the data.', default=False)
    args = parser.parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(level)

    try:
        if args.simulate or not args.plot:
            node_count = 1000
            nodes = list(itertools.islice(node_synthesizer(), node_count))
            bandwidth_graph = generate_bandwidth_graph(nodes)
            cluster_context = SimulationClusterContext(nodes, bandwidth_graph)

            # Run the skippy simulation
            scheduler = Scheduler(cluster_context)
            results_skippy = simulate(cluster_context, scheduler)
            results_skippy.to_csv('results/sim_skippy.csv')

            # Run the default scheduler simulation
            default_priorities: List[Tuple[float, Priority]] = [(1.0, BalancedResourcePriority()),
                                                                (1.0, ImageLocalityPriority())]
            scheduler = Scheduler(cluster_context=cluster_context,
                                  percentage_of_nodes_to_score=50,
                                  priorities=default_priorities)
            results_default = simulate(cluster_context, scheduler)
            results_default.to_csv('results/sim_default.csv')
        else:
            results_skippy = pd.DataFrame()
            results_skippy.from_csv('results/sim_skippy.csv')
            results_default = pd.DataFrame()
            results_default.from_csv('results/sim_default.csv')

        if args.plot or not args.simulate:
            plot_placement_time_cdf(results_skippy, 'skippy')
            plot_placement_time_cdf(results_default, 'default')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
