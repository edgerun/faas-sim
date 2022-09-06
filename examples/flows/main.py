import logging
import time

from examples.decentralized_loadbalancers.topology import testbed_topology
from examples.watchdogs.main import TrainInferenceBenchmark, AIFunctionSimulatorFactory
from sim.faassim import Simulation
from sim.factory.flow import SafeFlowFactory, UninterruptingFlowFactory, FlowFactory
from sim.requestgen import SimpleFunctionRequestFactory
from sim.util.client import find_clients

logger = logging.getLogger(__name__)


def execute_benchmark(flow_factory: FlowFactory):
    topology = testbed_topology()

    clients = find_clients(topology)

    #  inference factory - we assume that the file size is 250KB - client is a raspberry pi
    inference_factory = SimpleFunctionRequestFactory(client=clients[0].name, size=250)

    # training factory - we assume that the file size is 10MB (100000KB) - client is a raspberry pi
    train_factory = SimpleFunctionRequestFactory(client=clients[1].name, size=10000)

    benchmark = TrainInferenceBenchmark(inference_factory, train_factory)

    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(topology, benchmark)

    # override the SimulatorFactory factory
    sim.create_simulator_factory = AIFunctionSimulatorFactory

    # inject the flow factory
    sim.env.flow_factory = flow_factory

    # run the simulation
    start = time.time()
    sim.run()
    end = time.time()
    duration = end - start

    return duration, sim.env.metrics.extract_dataframe('flow')['duration'].describe()


def log_results(type: str, duration, results):
    logger.info(f'{type} flow simulation took {duration} seconds')
    logger.info(f'{type} flow simulation results... mean: {results["mean"]}, '
                f'std: {results["std"]}, '
                f'min: {results["min"]}, '
                f'25%: {results["25%"]}, '
                f'50%: {results["50%"]}, '
                f'75%: {results["75%"]}, '
                f'max: {results["max"]}, '
                f'count: {results["count"]}')


def main():
    # this simulation can use either the SafeFlow or UninterruptingFlow implementation
    # the SafeFlow will interrup all flows if a new flow gets created and sets the bandwidth accordingly
    # this should lead to a more accurate network simulation but leads to dramatic slowdown
    # The UninterruptingFlow implementation does not consider this and ignores newly created flows
    # which arguably leads to less accurate network simulations but increases speed of the simulation
    # The following example shows how to switch between the two types of simulations and runs both to compare the
    # wall clock runtime of the simulations.
    logging.basicConfig(level=logging.DEBUG)

    safe_flow_factory = SafeFlowFactory()
    uninterrupting_flow_factory = UninterruptingFlowFactory()

    safe_duration, safe_results = execute_benchmark(safe_flow_factory)
    uninterrupting_duration, uninterrupting_results = execute_benchmark(uninterrupting_flow_factory)

    log_results('uninterrupting', uninterrupting_duration, uninterrupting_results)
    log_results('safe', safe_duration, safe_results)


if __name__ == '__main__':
    main()
