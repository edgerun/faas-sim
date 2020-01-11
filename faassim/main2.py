import time

from sim.faas import FaasSimEnvironment
from sim.scenarios import Scenario, TestScenario, TestScenario2


class Simulation:

    def __init__(self, scenario: Scenario) -> None:
        super().__init__()
        self.scenario = scenario

        self.cluster = scenario.cluster()
        self.env = FaasSimEnvironment(self.cluster)
        self.env.process(self.env.faas_gateway.request_worker())
        self.env.process(self.env.faas_gateway.scheduler_worker())
        self.env.process(scenario.scenario_daemon(self.env))

    def run(self, until=None):
        env = self.env

        env.run(until=until)


def main():
    sim = Simulation(TestScenario2())
    then = time.time()
    sim.run(60 * 60 * 24)
    print('simulation took %.2f ms' % ((time.time() - then) * 1000))
    print('simulation time is now: %.2f' % sim.env.now)


if __name__ == '__main__':
    main()
