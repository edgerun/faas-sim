import concurrent.futures
import logging
import time

import srds.srds as stats
from skippy.core.model import Node
from skippy.core.priorities import BalancedResourcePriority, \
    LatencyAwareImageLocalityPriority, CapabilityPriority, DataLocalityPriority, LocalityTypePriority, \
    ImageLocalityPriority

from faassim.faas import BadPlacementException
from faassim.faassim import Simulation
from faassim.scenarios import UrbanSensingScenario, IndustrialIoTScenario, CloudRegionScenario

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

# iiot scenario weights
# weights = [8.29646, 1.54538, 0.62121, 9.67983, 6.96152]

# cloud region scenario weights
# weights = [0.65992, 6.92733, 0.73502, 5.81942, 0.6711]

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


def get_scenario_01():
    # 4 cells => 91 nodes

    return [
        UrbanSensingScenario(4, 9),
        UrbanSensingScenario(4, 18),
        UrbanSensingScenario(4, 27),
        UrbanSensingScenario(4, 36),
        UrbanSensingScenario(4, 46),
        UrbanSensingScenario(4, 55),
        UrbanSensingScenario(4, 64),
        UrbanSensingScenario(4, 73),
        UrbanSensingScenario(4, 82),
        UrbanSensingScenario(4, 91),
        UrbanSensingScenario(4, 100),
        UrbanSensingScenario(4, 109),
        UrbanSensingScenario(4, 118),
        UrbanSensingScenario(4, 127),
        UrbanSensingScenario(4, 137),
        UrbanSensingScenario(4, 146),
        UrbanSensingScenario(4, 155),
        UrbanSensingScenario(4, 164),
        UrbanSensingScenario(4, 173),
        UrbanSensingScenario(4, 182)
    ]


def get_scenario_02():
    # 10 premises = 120 nodes
    return [
        IndustrialIoTScenario(10, 12),
        IndustrialIoTScenario(10, 24),
        IndustrialIoTScenario(10, 36),
        IndustrialIoTScenario(10, 48),
        IndustrialIoTScenario(10, 60),
        IndustrialIoTScenario(10, 72),
        IndustrialIoTScenario(10, 84),
        IndustrialIoTScenario(10, 96),
        IndustrialIoTScenario(10, 108),
        IndustrialIoTScenario(10, 120),
        IndustrialIoTScenario(10, 132),
        IndustrialIoTScenario(10, 144),
        IndustrialIoTScenario(10, 156),
        IndustrialIoTScenario(10, 168),
        IndustrialIoTScenario(10, 180),
        IndustrialIoTScenario(10, 192),
        IndustrialIoTScenario(10, 204),
        IndustrialIoTScenario(10, 216),
        IndustrialIoTScenario(10, 228),
        IndustrialIoTScenario(10, 240),
    ]


def get_scenario_03():
    # 3 regions * 150 = 450 nodes
    return [
        CloudRegionScenario(150, 45),
        CloudRegionScenario(150, 90),
        CloudRegionScenario(150, 135),
        CloudRegionScenario(150, 180),
        CloudRegionScenario(150, 225),
        CloudRegionScenario(150, 270),
        CloudRegionScenario(150, 315),
        CloudRegionScenario(150, 360),
        CloudRegionScenario(150, 405),
        CloudRegionScenario(150, 450),
        CloudRegionScenario(150, 495),
        CloudRegionScenario(150, 540),
        CloudRegionScenario(150, 585),
        CloudRegionScenario(150, 630),
        CloudRegionScenario(150, 675),
        CloudRegionScenario(150, 720),
        CloudRegionScenario(150, 765),
        CloudRegionScenario(150, 810),
        CloudRegionScenario(150, 855),
        CloudRegionScenario(150, 900),
    ]


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info('initializing scenarios')
    scenarios = get_scenario_02()

    sched_params = {
        'skippy': skippy_params,
        'skippyopt': skippy_params_opt,
        'kube50': kube_params_50,
        'kube100': kube_params_100,
    }

    arguments = []

    for scheduler, scheduler_params in sched_params.items():
        for scenario in scenarios:
            scenario_name = scenario.__class__.__name__
            num_deployments = scenario.max_deployments
            num_nodes = len([node for node in scenario.topology().nodes if isinstance(node, Node)])
            ratio = round(num_deployments / num_nodes, 1)

            logger.info('deployment %d, nodes %d (%.2f)', num_deployments, num_nodes, ratio)

            # with faas idler
            arguments.append((scenario, scheduler_params, True, f'depl_{scenario_name}_{scheduler}_idler_{ratio}'))
            # without
            arguments.append((scenario, scheduler_params, False, f'depl_{scenario_name}_{scheduler}_noidler_{ratio}'))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(run_sim, arguments):
            pass


if __name__ == '__main__':
    main()
