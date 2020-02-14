import concurrent.futures
import logging
import time

from core.model import Node
from core.priorities import BalancedResourcePriority, \
    LatencyAwareImageLocalityPriority, CapabilityPriority, DataLocalityPriority, LocalityTypePriority, \
    ImageLocalityPriority
from sim import stats
from sim.faas import BadPlacementException
from sim.faassim import Simulation
from sim.scenarios import UrbanSensingScenario

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

weights = [6.45377, 4.78474, 8.99672, 8.92804, 1.23396]
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


def run_sim(args):
    stats.seed(123)

    scenario = args[0]
    scheduler_parameters = args[1]
    faas_idler = args[2]
    data_prefix = args[3].rstrip('_')

    logging.info('starting simulation %s with parameters %s', data_prefix, scheduler_parameters)
    sim = Simulation(scenario, scheduler_parameters, faas_idler)
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

    # 4 cells => 87 nodes
    logger.info('initializing scenarios')

    scenarios = [
        UrbanSensingScenario(4, 4),
        UrbanSensingScenario(4, 9),
        UrbanSensingScenario(4, 13),
        UrbanSensingScenario(4, 35),
        UrbanSensingScenario(4, 44),
        UrbanSensingScenario(4, 52),
        UrbanSensingScenario(4, 61),
        UrbanSensingScenario(4, 70),
        UrbanSensingScenario(4, 78),
        UrbanSensingScenario(4, 87),
        UrbanSensingScenario(4, 96),
        UrbanSensingScenario(4, 104),
        UrbanSensingScenario(4, 113),
        UrbanSensingScenario(4, 122),
        UrbanSensingScenario(4, 131),
        UrbanSensingScenario(4, 139),
        UrbanSensingScenario(4, 148),
        UrbanSensingScenario(4, 157),
        UrbanSensingScenario(4, 165),
        UrbanSensingScenario(4, 174)
    ]

    sched_params = {
        'skippy': skippy_params,
        'skippyopt': skippy_params_opt,
        'kube50': kube_params_50,
        'kube100': kube_params_100,
    }

    arguments = []

    for scheduler, scheduler_params in sched_params.items():
        for scenario in scenarios:
            num_deployments = scenario.max_deployments
            num_nodes = len([node for node in scenario.topology().nodes if isinstance(node, Node)])
            ratio = round(num_deployments / num_nodes, 1)

            logger.info('deployment %d, nodes %d (%.2f)', num_deployments, num_nodes, ratio)

            # with faas idler
            arguments.append((scenario, scheduler_params, True, f'depl_{scheduler}_idler_{ratio}'))
            # without
            arguments.append((scenario, scheduler_params, False, f'depl_{scheduler}_noidler_{ratio}'))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(run_sim, arguments):
            pass


if __name__ == '__main__':
    main()
