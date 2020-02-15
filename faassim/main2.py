import concurrent.futures
import logging
import time

from core.priorities import BalancedResourcePriority, \
    LatencyAwareImageLocalityPriority, CapabilityPriority, DataLocalityPriority, LocalityTypePriority, \
    ImageLocalityPriority
from sim import stats
from sim.faas import BadPlacementException
from sim.faassim import Simulation
from sim.scenarios import CloudRegionScenario

logger = logging.getLogger(__name__)

skippy_params = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, LatencyAwareImageLocalityPriority()),
        (1, LocalityTypePriority()),
        (1, DataLocalityPriority()),
        (1, CapabilityPriority())
    ],
    'percentage_of_nodes_to_score': 100
}

# urban sensing scenario weights
weights = [6.66109, 2.77657, 6.69114, 8.47306, 1.06714]

skippy_params_opt = {
    'priorities': [
        (weights[0], BalancedResourcePriority()),
        (weights[1], LatencyAwareImageLocalityPriority()),
        (weights[2], LocalityTypePriority()),
        (weights[3], DataLocalityPriority()),
        (weights[4], CapabilityPriority())
    ],
    'percentage_of_nodes_to_score': 100
}

kube_params_50 = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
    ],
    'percentage_of_nodes_to_score': 50
}

kube_params_100 = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
    ],
    'percentage_of_nodes_to_score': 100
}

Scenario = CloudRegionScenario


def run_sim(args):
    stats.seed(123)

    scheduler_parameters = args[0]
    faas_idler = args[1]
    data_prefix = args[2].rstrip('_')

    logging.info('starting simulation %s with parameters %s', data_prefix, scheduler_parameters)
    sim = Simulation(Scenario.lazy(), scheduler_parameters, faas_idler)
    then = time.time()
    try:
        sim.run()
    except BadPlacementException as e:
        logger.error('Could not finish simulation %s', data_prefix)
        return data_prefix

    sim.dump_data_frames('/tmp/schedsim', prefix=data_prefix + '_')
    logging.info('simulation %s took %.2f ms', data_prefix, ((time.time() - then) * 1000))
    return data_prefix


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info('initializing scenario')
    Scenario.lazy(vms_per_region=150)

    params = {
        'skippy': skippy_params,
        'skippyopt': skippy_params_opt,
        'kube50': kube_params_50,
        'kube100': kube_params_100,
    }
    runs = 1

    arguments = []
    # with faas idler
    arguments.extend([(p, True, f'{k}_idler_{i + 1:03}') for i in range(runs) for k, p in params.items()])
    # without
    arguments.extend([(p, False, f'{k}_noidler_{i + 1:03}') for i in range(runs) for k, p in params.items()])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(run_sim, arguments):
            pass

    Scenario.purge()


if __name__ == '__main__':
    main()
