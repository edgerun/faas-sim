import logging
import os
import time

import pandas as pd

from sim.faas import FaasSimEnvironment
from sim.scenarios import Scenario

logger = logging.getLogger(__name__)


class SimulationTimeoutError(BaseException):
    pass


def _timeout_listener(env, started, max_time, interval=1):
    while True:
        yield env.timeout(interval)

        if (time.time() - started) > max_time:
            raise SimulationTimeoutError()


class Simulation:

    def __init__(self, scenario: Scenario, scheduler_params: dict = None, faas_idler=True) -> None:
        super().__init__()
        self.scenario = scenario

        self.scheduler_params = scheduler_params or scenario.scheduler_parameters()

        self.env = FaasSimEnvironment(scenario.topology(), scheduler_params=self.scheduler_params)
        self.env.process(self.env.faas_gateway.request_worker())
        self.env.process(self.env.faas_gateway.scheduler_worker())
        if faas_idler:
            self.env.process(self.env.faas_gateway.faas_idler())
        self.scenario_process = self.env.process(scenario.scenario_daemon(self.env))

        self._data_frames = dict()

    def run(self, until=None, timeout=None):
        logger.info('simulation starting %s', self)

        env = self.env
        then = time.time()

        if timeout:
            env.process(_timeout_listener(env, then, timeout))

        if until is None:
            until = self.scenario_process

        env.run(until=until)

        logger.info('simulation %s finished in %.2f seconds', self, (time.time() - then))

    def dataframe(self, metric):
        if metric not in self._data_frames:
            df = self.env.metrics.extract_dataframe(metric)
            self._data_frames[metric] = df

        return self._data_frames[metric]

    def dump_data_frames(self, directory='/tmp', prefix=None):
        prefix = prefix or 'schedsim_%s_' % time.strftime('%Y-%m-%d-%H-%M-%S')

        metrics = {record.measurement for record in self.env.metrics.records}

        for metric in metrics:
            self.dataframe(metric).to_csv(os.path.join(directory, f'{prefix}{metric}.csv'))

        # add dataframe containing all nodes
        df = pd.DataFrame([{
            'name': node.name,
            'memory': node.capacity.memory,
            'cpu': node.capacity.cpu_millis,
            'labels': node.labels
        } for node in self.env.cluster.list_nodes()])
        df.index = df['name']
        del df['name']
        df.to_csv(os.path.join(directory, f'{prefix}nodes.csv'))
