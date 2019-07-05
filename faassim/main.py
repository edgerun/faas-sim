import time
import simpy
import logging

from core.scheduler import Scheduler
from sim.simclustercontext import SimulationClusterContext
from sim.stats import exp_sampler
from sim.synth import pod_synthesizer, PodSynthesizer


def run_load_generator(env: simpy.Environment, queue: simpy.Store, pod_synth: PodSynthesizer, ia_sampler):
    """
    :param env: simpy environment
    :param queue: the work queue
    :param pod_synth: fake Pod generator
    :param ia_sampler: arrival profile
    :return:
    """
    while True:
        ia = next(ia_sampler)  # inter-arrival
        ia = round(ia, 3)  # millisecond accuracy
        yield env.timeout(ia)

        pod = next(pod_synth)
        queue.put(pod)

        logging.debug('pod arrived at %.2f seconds' % env.now)


def run_scheduler_worker(env: simpy.Environment, queue: simpy.Store, scheduler: Scheduler):
    while True:
        logging.debug('Scheduler waiting for pod...')
        pod = yield queue.get()

        # TODO fix time not changing (env.now)
        logging.debug('Pod received by scheduler at %.2f', env.now)
        then = time.time()

        # execute scheduling algorithm
        result = scheduler.schedule(pod)

        duration = ((time.time() - then) * 1000)
        yield env.timeout(duration / 1000)
        logging.debug('Pod scheduling took %.2f ms, and yielded %s', duration, result)


def main():
    # TODO
    # - Properly synthesize ml-wf-workload
    # - Make sure to measure the placement quality instead of the placement time
    logging.getLogger().setLevel(logging.DEBUG)

    cluster_context = SimulationClusterContext()
    scheduler = Scheduler(cluster_context)

    env = simpy.RealtimeEnvironment(factor=0.01, strict=False)
    queue = simpy.Store(env)
    env.process(run_load_generator(env, queue, pod_synthesizer(), exp_sampler(lambd=1.5)))
    env.process(run_scheduler_worker(env, queue, scheduler))
    env.sync()
    try:
        env.run(until=200)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
