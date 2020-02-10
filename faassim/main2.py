import logging
import time

from sim.faassim import Simulation
from sim.scenarios import TestScenario


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
