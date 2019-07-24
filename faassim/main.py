import argparse
import ast
import itertools
import time
from typing import List, Generator, Tuple
import simpy
import pandas as pd
import logging

from core.clustercontext import ClusterContext
from core.priorities import Priority, BalancedResourcePriority, ImageLocalityPriority
from core.scheduler import Scheduler
from sim.model import EventType, LoggingRow
from sim.oracle.oracle import StartupTimeOracle, ExecutionTimeOracle, Oracle, BandwidthUsageOracle, CostOracle, \
    ResourceUtilizationOracle
from sim.plotting import plot_startup_time_cdf, plot_execution_times_boxplot, plot_startup_times_boxplot, \
    plot_task_completion_times, plot_combined_startup_time_cdf, plot_execution_times_bar, plot_startup_times_bar
from sim.simclustercontext import SimulationClusterContext
from sim.stats import exp_sampler
from sim.synth.bandwidth import generate_bandwidth_graph
from sim.synth.nodes import node_synthesizer
from sim.synth.pods import pod_synthesizer, PodSynthesizer


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

        # weight the startup
        metadata = dict([o.estimate(context, pod, result) for o in oracles])

        # also add the image name to the metadata
        metadata['image'] = pod.spec.containers[0].image

        log.append(LoggingRow(env.now, EventType.POD_SCHEDULED, pod.name, metadata))


def simulate(cluster_context: ClusterContext, scheduler: Scheduler) -> pd.DataFrame:
    log = []
    oracles = [StartupTimeOracle(),
               ExecutionTimeOracle(),
               BandwidthUsageOracle(),
               CostOracle(),
               ResourceUtilizationOracle()]
    env = simpy.RealtimeEnvironment(factor=0.01, strict=False)
    queue = simpy.Store(env)
    env.process(run_load_generator(env, queue, pod_synthesizer(), exp_sampler(lambd=1.5), log))
    env.process(run_scheduler_worker(env, queue, cluster_context, scheduler, oracles, log))
    env.sync()
    env.run(until=1000)
    data = pd.DataFrame(data=log)
    return data


def read_csv(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df['event'] = df['event'].apply(lambda x: EventType[x[10:]])
    df['additional_attributes'] = df['additional_attributes'].apply(lambda x: ast.literal_eval(x))
    return df


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
            logging.info('Starting the simulation...')
            node_count = 1000
            nodes = list(itertools.islice(node_synthesizer(), node_count))
            bandwidth_graph = generate_bandwidth_graph(nodes)
            cluster_context = SimulationClusterContext(nodes, bandwidth_graph)

            # Run the skippy simulation
            scheduler = Scheduler(cluster_context)
            logging.info('Simulating the Skippy scheduler...')
            results_skippy = simulate(cluster_context, scheduler)
            results_skippy.to_csv('results/sim_skippy.csv')

            # Run the default scheduler simulation
            logging.info('Simulating the default scheduler...')
            # TODO make sure these priority list contains all influencing prio functions
            default_priorities: List[Tuple[float, Priority]] = [(1.0, BalancedResourcePriority()),
                                                                (1.0, ImageLocalityPriority())]
            cluster_context = SimulationClusterContext(nodes, bandwidth_graph)
            scheduler = Scheduler(cluster_context=cluster_context,
                                  percentage_of_nodes_to_score=50,
                                  priorities=default_priorities)
            results_default = simulate(cluster_context, scheduler)
            results_default.to_csv('results/sim_default.csv')
            logging.info('Simulation finished.')
        else:
            logging.info('Loading pre-recorded simulation data...')
            results_default = read_csv('results/sim_default.csv')
            results_skippy = read_csv('results/sim_skippy.csv')
            logging.info('Simulation data loaded.')

        if args.plot or not args.simulate:
            logging.info('Plotting data...')
            plot_startup_time_cdf(results_default, 'default')
            plot_startup_time_cdf(results_skippy, 'skippy')
            plot_combined_startup_time_cdf([('default', results_default), ('skippy', results_skippy)])
            plot_startup_times_bar(results_default, results_skippy)
            plot_execution_times_bar(results_default, results_skippy)
            plot_execution_times_boxplot(results_default, results_skippy)
            plot_startup_times_boxplot(results_default, results_skippy)
            plot_task_completion_times(results_default, results_skippy)

        logging.info('Done.')
    except KeyboardInterrupt:
        logging.info('Aborting after KeyboardInterrupt.')
        pass


if __name__ == '__main__':
    main()
