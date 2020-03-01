import ast
import logging
import time
from typing import List, Generator

import pandas as pd
import simpy
from skippy.core.clustercontext import ClusterContext
from skippy.core.scheduler import Scheduler
from srds import ParameterizedDistribution as PDist, BufferedSampler

from faassim.model import EventType, LoggingRow
from faassim.oracle.oracle import Oracle, BandwidthUsageOracle, CostOracle, ResourceUtilizationOracle, \
    FittedExecutionTimeOracle, FittedStartupTimeOracle
from faassim.synth.pods import PodSynthesizer


def exp_sampler(lambd):
    sampler = BufferedSampler(PDist.expon(lambd))
    while True:
        yield sampler.sample()


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

        # execute scheduling algorithm
        then = time.time()
        result = scheduler.schedule(pod)
        duration = time.time() - then

        yield env.timeout(duration)

        logging.debug('Pod scheduling took %.2f ms, and yielded %s', duration * 1000, result)

        # weight the startup
        metadata = dict([o.estimate(context, pod, result) for o in oracles])

        # also add the image name and the selected node to the metadata
        metadata['image'] = pod.spec.containers[0].image
        metadata['suggested_host'] = None if result.suggested_host is None else result.suggested_host.name

        log.append(LoggingRow(env.now, EventType.POD_SCHEDULED, pod.name, metadata))


def simulate(cluster_context: ClusterContext, scheduler: Scheduler, pod_synth: PodSynthesizer,
             until: int) -> pd.DataFrame:
    log = []
    oracles = [
        FittedStartupTimeOracle(),
        FittedExecutionTimeOracle(),
        BandwidthUsageOracle(),
        CostOracle(FittedExecutionTimeOracle()),
        ResourceUtilizationOracle()
    ]
    env = simpy.Environment()
    queue = simpy.Store(env)
    env.process(run_load_generator(env, queue, pod_synth, exp_sampler(lambd=1.5), log))
    env.process(run_scheduler_worker(env, queue, cluster_context, scheduler, oracles, log))
    env.run(until=until)

    data = pd.DataFrame(data=log)

    return data


def read_csv(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df['event'] = df['event'].apply(lambda x: EventType[x[10:]])
    df['additional_attributes'] = df['additional_attributes'].apply(lambda x: ast.literal_eval(x))
    return df
