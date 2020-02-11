import concurrent.futures
import logging
import time

from core.priorities import BalancedResourcePriority, \
    LatencyAwareImageLocalityPriority, CapabilityPriority, DataLocalityPriority, LocalityTypePriority, \
    ImageLocalityPriority
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
    scheduler_parameters = args[0]
    faas_idler = args[1]
    data_prefix = args[2].rstrip('_')

    logging.info('starting simulation %s with parameters %s', data_prefix, scheduler_parameters)
    sim = Simulation(UrbanSensingScenario(), scheduler_parameters, faas_idler)
    then = time.time()
    sim.run()
    sim.dump_data_frames('/tmp/schedsim', prefix=data_prefix + '_')
    logging.info('simulation %s took %.2f ms', data_prefix, ((time.time() - then) * 1000))
    return data_prefix


def main():
    logging.basicConfig(level=logging.INFO)

    params = {
        'skippy': skippy_params,
        'kube50': kube_params_50,
        'kube100': kube_params_100,
    }
    runs = 10

    arguments = []
    # with faas idler
    arguments.extend([(p, True, f'{k}_idler_{i + 1:03}') for i in range(runs) for k, p in params.items()])
    # without
    arguments.extend([(p, False, f'{k}_noidler_{i + 1:03}') for i in range(runs) for k, p in params.items()])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _ in executor.map(run_sim, arguments):
            pass


if __name__ == '__main__':
    main()
