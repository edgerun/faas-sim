import argparse
import time
from tkinter import EventType
from typing import NamedTuple, Dict, List, Generator
import simpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from core.scheduler import Scheduler
from sim.simclustercontext import SimulationClusterContext
from sim.stats import exp_sampler
from sim.synth import pod_synthesizer, PodSynthesizer


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


def run_scheduler_worker(env: simpy.Environment, queue: simpy.Store, log: List[LoggingRow], scheduler: Scheduler):
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
        log.append(LoggingRow(env.now, EventType.POD_SCHEDULED, pod.name))


def simulate():
    log = []
    cluster_context = SimulationClusterContext()
    scheduler = Scheduler(cluster_context)

    env = simpy.RealtimeEnvironment(factor=0.01, strict=False)
    queue = simpy.Store(env)
    env.process(run_load_generator(env, queue, pod_synthesizer(), exp_sampler(lambd=1.5), log))
    env.process(run_scheduler_worker(env, queue, log, scheduler))
    env.sync()
    try:
        env.run(until=200)
        data = pd.DataFrame(data=log)
        # TODO implement oracle to calculate execution and placement time for a selected node / config
        #  Add the calucalated execution time and placement time via the oracle and add it to the log rows
        # TODO save the CSV
    except KeyboardInterrupt:
        pass


def plot():
    # TODO plot the data
    pass


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Skippy Simulator')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug',
                        help='Enable debug logs.', default=False)
    parser.add_argument('-s', '--simulate', action='store_true', dest='debug',
                        help='Only simulate the scheduling.', default=False)
    parser.add_argument('-p', '--plot', action='store_true', dest='debug',
                        help='Only plot the data.', default=False)
    args = parser.parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(level)
    
    if args.simulate or not args.plot:
        simulate()

    if args.plot or not args.simulate:
        plot()


if __name__ == '__main__':
    main()
