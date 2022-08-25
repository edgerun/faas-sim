import logging
import math
from typing import List

import numpy as np
from faas.system.core import FunctionReplicaState, FaasSystem

from .core import Environment
from .system import SimFunctionDeployment, SimFunctionReplica
from .watchdogs import HTTPWatchdog

logger = logging.getLogger(__name__)


def faas_idler(env: Environment, inactivity_duration=300, reconcile_interval=30):
    """
    https://github.com/openfaas-incubator/faas-idler
    https://github.com/openfaas-incubator/faas-idler/blob/master/main.go

    default values:
    https://github.com/openfaas-incubator/faas-idler/blob/668991c532156275993399ee79a297a4c2d651ec/docker-compose.yml

    :param env: the faas environment
    :param inactivity_duration: i.e. 15m (Golang duration)
    :param reconcile_interval: i.e. 1m (default value)
    :return: an event generator
    """
    faas: FaasSystem = env.faas
    while True:
        yield env.timeout(reconcile_interval)

        for deployment in faas.get_deployments():
            if not deployment.scaling_configuration.scale_zero:
                continue

            name = deployment.name
            replicas = faas.get_replicas(name, running=True)
            if len(replicas) == 0:
                continue

            idle_time = env.now - env.metrics.last_invocation[name]
            if idle_time >= inactivity_duration:
                env.process(faas.scale_down(name, replicas))
                logger.debug('%.2f function %s has been idle for %.2fs', env.now, name, idle_time)


class FaasRequestScaler:

    def __init__(self, fn: SimFunctionDeployment, env: Environment):
        self.env = env
        self.function_invocations = dict()
        self.reconcile_interval = fn.scaling_config.rps_threshold_duration
        self.threshold = fn.scaling_config.rps_threshold
        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: FaasSystem = env.faas
        while self.running:
            yield env.timeout(self.reconcile_interval)
            if self.function_invocations.get(self.fn_name, None) is None:
                self.function_invocations[self.fn_name] = 0
            last_invocations = self.function_invocations.get(self.fn_name, 0)
            current_total_invocations = env.metrics.invocations.get(self.fn_name, 0)
            invocations = current_total_invocations - last_invocations
            self.function_invocations[self.fn_name] += invocations
            # TODO divide by alert window, but needs to store the invocations, such that reconcile_interval != alert_window is possible
            config = self.fn.scaling_config
            if (invocations / self.reconcile_interval) >= self.threshold:
                scale = (config.scale_factor / 100) * config.scale_max
                yield from faas.scale_up(self.fn_name, int(scale))
                logger.debug(f'scaled up {self.fn_name} by {scale}')
            else:
                scale = (config.scale_factor / 100) * config.scale_max
                yield from faas.scale_down(self.fn_name, int(scale))
                logger.debug(f'scaled down {self.fn_name} by {scale}')

    def stop(self):
        self.running = False


class AverageFaasRequestScaler:
    """
    Scales deployment according to the average number of requests distributed equally over all replicas.
    The distributed property holds as per default the round robin scheduler is used
    """

    def __init__(self, fn: SimFunctionDeployment, env: Environment):
        self.env = env
        self.function_invocations = dict()
        self.threshold = fn.scaling_config.target_average_rps
        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: FaasSystem = env.faas
        while self.running:
            yield env.timeout(self.alert_window)
            if self.function_invocations.get(self.fn_name, None) is None:
                self.function_invocations[self.fn_name] = 0
            running_replicas = faas.get_replicas(self.fn.name, True)
            running = len(running_replicas)
            if running == 0:
                continue

            conceived_replicas = faas.get_replicas(self.fn.name, state=FunctionReplicaState.CONCEIVED)
            pending_replicas = faas.get_replicas(self.fn.name, state=FunctionReplicaState.PENDING)

            last_invocations = self.function_invocations.get(self.fn_name, 0)
            current_total_invocations = env.metrics.invocations.get(self.fn_name, 0)
            invocations = current_total_invocations - last_invocations
            self.function_invocations[self.fn_name] += invocations
            average = invocations / running
            desired_replicas = math.ceil(running * (average / self.threshold))

            updated_desired_replicas = desired_replicas
            if len(conceived_replicas) > 0 or len(pending_replicas) > 0:
                if desired_replicas > len(running_replicas):
                    count = len(running_replicas) + len(conceived_replicas) + len(pending_replicas)
                    average = invocations / count
                    updated_desired_replicas = math.ceil(running * (average / self.threshold))

            if desired_replicas > len(running_replicas) and updated_desired_replicas < len(running_replicas):
                # no scaling in case of reversed decision
                continue

            ratio = average / self.threshold
            if 1 > ratio >= 1 - self.fn.scaling_config.target_average_rps_threshold:
                # ratio is sufficiently close to 1.0
                continue

            if 1 < ratio < 1 + self.fn.scaling_config.target_average_rps_threshold:
                continue

            if desired_replicas < len(running_replicas):
                # scale down
                scale = len(running_replicas) - desired_replicas
                yield from faas.scale_down(self.fn.name, scale)
            else:
                # scale up
                scale = desired_replicas - len(running_replicas)
                yield from faas.scale_up(self.fn.name, scale)

    def stop(self):
        self.running = False


class AverageQueueFaasRequestScaler:
    """
    Scales deployment according to the average number of requests distributed equally over all replicas.
    The distributed property holds as per default the round robin scheduler is used
    """

    def __init__(self, fn: SimFunctionDeployment, env: Environment):
        self.env = env
        self.threshold = fn.scaling_config.target_queue_length
        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: 'DefaultFaasSystem' = env.faas
        while self.running:
            yield env.timeout(self.alert_window)
            running_replicas: List[SimFunctionReplica] = faas.get_replicas(self.fn.name, running=True)
            running = len(running_replicas)
            if running == 0:
                continue

            conceived_replicas = faas.get_replicas(self.fn.name, state=FunctionReplicaState.CONCEIVED)
            starting_replicas = faas.get_replicas(self.fn.name, state=FunctionReplicaState.PENDING)

            in_queue = []
            for replica in running_replicas:
                sim = replica.simulator
                if isinstance(sim, HTTPWatchdog):
                    in_queue.append(len(sim.queue.queue))
            if len(in_queue) == 0:
                average = 0
            else:
                average = int(math.ceil(np.median(np.array(in_queue))))

            desired_replicas = math.ceil(running * (average / self.threshold))

            updated_desired_replicas = desired_replicas
            if len(conceived_replicas) > 0 or len(starting_replicas) > 0:
                if desired_replicas > len(running_replicas):
                    for i in range(len(conceived_replicas) + len(starting_replicas)):
                        in_queue.append(0)

                    average = int(math.ceil(np.median(np.array(in_queue))))
                    updated_desired_replicas = math.ceil(running * (average / self.threshold))

            if desired_replicas > len(running_replicas) and updated_desired_replicas < len(running_replicas):
                # no scaling in case of reversed decision
                continue

            ratio = average / self.threshold
            if 1 > ratio >= 1 - self.fn.scaling_config.target_average_rps_threshold:
                # ratio is sufficiently close to 1.0
                continue

            if 1 < ratio < 1 + self.fn.scaling_config.target_average_rps_threshold:
                continue

            if desired_replicas < len(running_replicas):
                # scale down
                scale = len(running_replicas) - desired_replicas
                yield from faas.scale_down(self.fn.name, scale)
            else:
                # scale up
                scale = desired_replicas - len(running_replicas)
                yield from faas.scale_up(self.fn.name, scale)

    def stop(self):
        self.running = False
