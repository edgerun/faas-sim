import logging
import os
import time
from typing import List

import numpy as np
import pandas as pd

from sim.faas import FaasSimEnvironment
from sim.logging import Record
from sim.scenarios import Scenario, TestScenario


def extract_dataframe(measurement: str, records: List[Record]):
    data = list()

    for record in records:
        if record.measurement != measurement:
            continue

        r = dict()
        r['time'] = record.time
        for k, v in record.fields.items():
            r[k] = v
        for k, v in record.tags.items():
            r[k] = v

        data.append(r)

    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex(pd.to_datetime(df['time']))
    del df['time']
    return df


class Simulation:

    def __init__(self, scenario: Scenario) -> None:
        super().__init__()
        self.scenario = scenario

        self.env = FaasSimEnvironment(scenario.topology())
        self.env.process(self.env.faas_gateway.request_worker())
        self.env.process(self.env.faas_gateway.scheduler_worker())
        self.env.process(self.env.faas_gateway.faas_idler())
        self.env.process(scenario.scenario_daemon(self.env))

        self._data_frames = dict()

    def run(self, until=None):
        env = self.env
        env.run(until=until)

    def dataframe(self, metric):
        if metric not in self._data_frames:
            df = extract_dataframe(metric, self.env.metrics.records)
            self._data_frames[metric] = df

        return self._data_frames[metric]

    def dump_data_frames(self, directory='/tmp', prefix=None):
        prefix = prefix or 'schedsim_%s_' % time.strftime('%y-%m-%d-%H-%M')

        metrics = {record.measurement for record in self.env.metrics.records}

        for metric in metrics:
            self.dataframe(metric).to_csv(os.path.join(directory, f'{prefix}{metric}.csv'))

    def calc_total_network(self):
        return self.dataframe('network')['value'].sum() / 10e6

    def calc_average_fet(self):
        df = self.dataframe('invocations')
        df = df.dropna()
        df['rtt'] = df['t_exec'] + df['t_wait']

        return df['rtt'].mean()

    def calc_edge_utilization(self):
        df = self.dataframe('allocation')
        nodes = [node.name for node in self.env.cluster.list_nodes()]

        dfg = df.groupby('node').aggregate('mean')
        dfg = dfg.reindex(nodes, fill_value=0)
        dfg['zone'] = np.where(dfg.index.str.endswith('cloud'), 'cloud', 'edge')
        dfg = dfg.groupby('zone').mean()

        util_c = sum(dfg.loc['cloud'])
        util_e = sum(dfg.loc['edge'])

        return util_e / (util_e + util_c)

    def calc_cloud_cost(self):
        # TODO https://aws.amazon.com/lambda/pricing/
        # https://aws.amazon.com/ec2/pricing/on-demand/ (Data Transfer)

        # seconds = np.ceil(seconds + 0.05)
        # total_compute_seconds = sum(np.ceil(seconds + 0.05))
        # compute_gb_seconds = max(128, min(3072, ram)) * np.ceil(seconds + 0.05) # how much ram was allocated during execution
        # compute_usd = compute_gb_seconds * 0.0000166667
        # requests = count(invocations)
        # request_usd = requests * 0.0000002

        # return compute_usd * request_usd

        raise NotImplementedError


def main():
    logging.basicConfig(level=logging.INFO)
    sim = Simulation(TestScenario.lazy())
    then = time.time()
    sim.run(60 * 60)
    print('simulation took %.2f ms' % ((time.time() - then) * 1000))
    print('simulation time is now: %.2f' % sim.env.now)

    sim.dump_data_frames()


if __name__ == '__main__':
    main()
