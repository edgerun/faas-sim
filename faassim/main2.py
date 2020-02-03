import logging
import time
from typing import List

import matplotlib.pyplot as plt
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

    def run(self, until=None):
        env = self.env
        env.run(until=until)

        print(env.metrics.invocations)

        df = extract_dataframe('invocations', self.env.metrics.records)

        x = df['t_exec'].resample('10s').max()
        plt.plot(x)
        x = df['t_wait'].resample('10s').max()
        plt.plot(x)
        plt.show()

        df['function_type'] = df['function_name'].str.split('_').str[2]
        x = df[['function_type', 't_wait', 't_exec']]

        counts = x.groupby('function_type')

        for key, grp in counts:
            y = grp['t_exec'].resample('60s').count()
            plt.plot(y, label=key)
        plt.title('average requests/minute')
        plt.legend()
        plt.show()

        groups = x.groupby('function_type').mean()[['t_wait', 't_exec']]
        print(groups)

        df_alloc = extract_dataframe('allocation', self.env.metrics.records)
        df_alloc = df_alloc[df_alloc['node'] == 'edge_9_2_tegra']
        y = df_alloc['cpu']
        plt.step(y.index, y)
        plt.show()

        y = df_alloc['mem']
        plt.step(y.index, y)
        plt.show()

        df_util = extract_dataframe('utilization', self.env.metrics.records)
        df_util = df_util[df_util['node'] == 'edge_9_2_tegra']
        y = df_util['mem'].resample('10s').mean().replace(np.nan, 0)
        plt.ylim(0, 1)
        plt.plot(y)
        plt.show()


def main():
    logging.basicConfig(level=logging.INFO)
    sim = Simulation(TestScenario.lazy())
    then = time.time()
    sim.run(60 * 60)
    print('simulation took %.2f ms' % ((time.time() - then) * 1000))
    print('simulation time is now: %.2f' % sim.env.now)


if __name__ == '__main__':
    main()
