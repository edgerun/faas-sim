import time
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from sim.faas import FaasSimEnvironment
from sim.logging import Record
from sim.scenarios import Scenario, TestScenario2


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

        self.cluster = scenario.cluster()
        self.env = FaasSimEnvironment(self.cluster)
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

        x = df['t_wait'].resample('60s').count()
        plt.plot(x)
        plt.show()

        df['function_type'] = df['function_name'].str.split('_').str[2]
        x = df[['function_type', 't_wait', 't_exec']]

        groups = x.groupby('function_type').mean()[['t_wait', 't_exec']]
        print(groups)

        df = extract_dataframe('allocation', self.env.metrics.records)
        df['node_name'] = df['node'].astype(str)
        df =df[df['node_name'] == '1_cloud']
        y = df['cpu']
        plt.step(y.index, y)
        plt.show()

        y = df['mem']
        plt.step(y.index, y)
        plt.show()



def main():
    sim = Simulation(TestScenario2())
    then = time.time()
    sim.run(60 * 60 * 3)
    print('simulation took %.2f ms' % ((time.time() - then) * 1000))
    print('simulation time is now: %.2f' % sim.env.now)


if __name__ == '__main__':
    main()
