import argparse
import concurrent.futures
import logging
import os
import sys
import time

import srds.srds as stats
from skippy.core.priorities import BalancedResourcePriority, \
    LatencyAwareImageLocalityPriority, CapabilityPriority, DataLocalityPriority, LocalityTypePriority

from faassim.faas import BadPlacementException
from faassim.faassim import Simulation
from faassim.scenarios import CloudRegionScenario, UrbanSensingScenario, IndustrialIoTScenario

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='The scenario to execute (1 = UrbanSensing, 2 = IIoT, 3 = CloudRegions)',
                        type=int, required=True)
    parser.add_argument('--results', help='directory to store results into (defaults to /tmp/schedsim)', type=str,
                        default='/tmp/schedsim')
    parser.add_argument('--logging', help='set log level (DEBUG|INFO|WARN|...) to activate logging', required=False)

    return parser.parse_args()


scheduler_params = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, LatencyAwareImageLocalityPriority()),
        (1, LocalityTypePriority()),
        (1, DataLocalityPriority()),
        (1, CapabilityPriority())
    ],
    'percentage_of_nodes_to_score': 100
}

scenarios = [
    (UrbanSensingScenario, (4,), {}),
    (IndustrialIoTScenario, (5,), {}),
    (CloudRegionScenario, (50,), {})
]


def run_sim(scenario, scheduler_parameters, faas_idler, data_prefix, results_dir):
    stats.seed(0)

    logging.info('starting simulation %s with parameters %s', data_prefix, scheduler_parameters)
    sim = Simulation(scenario, scheduler_parameters, faas_idler)
    then = time.time()
    try:
        sim.run()
    except BadPlacementException as e:
        logger.error('Could not finish simulation %s', data_prefix)
        return data_prefix

    sim.dump_data_frames(results_dir, prefix=data_prefix + '_')
    logging.info('simulation %s took %.2f ms', data_prefix, ((time.time() - then) * 1000))
    return data_prefix


def main():
    args = parse_args()

    if args.logging:
        logging.basicConfig(level=logging._nameToLevel[args.logging])

    if not os.path.exists(args.results):
        logger.info('creating results directory %s', args.results)
        os.makedirs(args.results, exist_ok=True)
    if not os.path.isdir(args.results):
        print('expected %s to be a directory' % args.results, file=sys.stderr)
        exit(1)

    logger.info('initializing scenario %d', args.scenario)
    scenario_class, scenario_args, scenario_kwargs = scenarios[args.scenario]
    scenario = scenario_class.lazy(*scenario_args, **scenario_kwargs)

    try:
        ts = time.strftime('%Y%m%d%H%M')
        name = f'skippy_noidler_{args.scenario}-{ts}'
        run_sim(scenario, scheduler_params, False, name, args.results)
    finally:
        scenario_class.purge()


if __name__ == '__main__':
    main()
