import logging

import examples.basic.main as basic
from examples.custom_function_sim.main import CustomSimulatorFactory
from sim.faassim import Simulation

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(basic.example_topology(), basic.ExampleBenchmark())

    # override the SimulatorFactory factory
    sim.create_simulator_factory = CustomSimulatorFactory

    # run the simulation
    sim.run()

    dfs = {
        'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
        'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
        'scale_df': sim.env.metrics.extract_dataframe('scale'),
        'schedule_df': sim.env.metrics.extract_dataframe('schedule'),
        'replica_deployment_df': sim.env.metrics.extract_dataframe('replica_deployment'),
        'function_deployments_df': sim.env.metrics.extract_dataframe('function_deployments'),
        'function_deployment_df': sim.env.metrics.extract_dataframe('function_deployment'),
        'function_deployment_lifecycle_df': sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
        'functions_df': sim.env.metrics.extract_dataframe('functions'),
        'flow_df': sim.env.metrics.extract_dataframe('flow'),
        'network_df': sim.env.metrics.extract_dataframe('network'),
        'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
        'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization'),
        'fets_df': sim.env.metrics.extract_dataframe('fets')
    }

    logger.info('Mean exec time %d', dfs['invocations_df']['t_exec'].mean())


if __name__ == '__main__':
    main()
