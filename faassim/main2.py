import logging
import os
import time

import numpy as np
import pandas as pd

from sim.faas import FaasSimEnvironment
from sim.scenarios import Scenario, TestScenario


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

    def calc_total_network(self):
        df = self.dataframe('network')
        df = df[(df.type == 'uplink') | (df.type == 'downlink')]
        return df['value'].sum() / 10e6

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
        price_per_gbsec = 0.000016667
        price_per_m_invocations = 0.2
        price_per_egress_gb = 0.07  # average

        df = self.dataframe('invocations')
        df['zone'] = np.where(df.node.str.contains('cloud'), 'cloud', 'edge')
        df = df[df['zone'] == 'cloud']

        sec = np.ceil(df['t_exec'] * 10) / 10  # round up to the nearest 100ms
        gb = df['memory'] / 10e9

        sum_compute = sum(sec * gb) * price_per_gbsec
        sum_invocations = np.ceil(len(df) / 10e6) * price_per_m_invocations  # million invocations

        df = self.dataframe('network')
        df = df[(df['zone'] == 'cloud') & (df['type'] == 'uplink')]
        egress_gb = df['value'] / 10e6

        sum_network = sum(egress_gb) * price_per_egress_gb

        return sum_compute + sum_invocations + sum_network


def main():
    logging.basicConfig(level=logging.INFO)
    sim = Simulation(TestScenario.lazy())
    then = time.time()
    sim.run(60 * 60)
    print('simulation took %.2f ms' % ((time.time() - then) * 1000))
    print('simulation time is now: %.2f' % sim.env.now)

    sim.dump_data_frames('/tmp/schedsim')

    print('calc_total_network', sim.calc_total_network())
    print('calc_average_fet', sim.calc_average_fet())
    print('calc_edge_utilization', sim.calc_edge_utilization())
    print('calc_cloud_cost', sim.calc_cloud_cost())


if __name__ == '__main__':
    main()
