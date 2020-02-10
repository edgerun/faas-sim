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

kube_params = {
    'priorities': [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
    ],
    'percentage_of_nodes_to_score': 50
}


def run_sim(args):
    scheduler_parameters = args[0]
    data_prefix = args[1]

    logging.info('starting simulation %s with parameters %s', data_prefix, scheduler_parameters)
    sim = Simulation(UrbanSensingScenario.lazy(), scheduler_parameters)
    then = time.time()
    sim.run(60 * 60)
    sim.dump_data_frames('/tmp/schedsim', prefix=data_prefix)
    logging.info('simulation %s took %.2f ms', data_prefix, ((time.time() - then) * 1000))
    return data_prefix


def main():
    logging.basicConfig(level=logging.INFO)

    params = {
        'skippy': skippy_params,
        'kube': kube_params
    }
    runs = 10

    arguments = [(p, f'{k}_{i + 1:03}_') for i in range(runs) for k, p in params.items()]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for _ in executor.map(run_sim, arguments):
            pass


if __name__ == '__main__':
    main()
