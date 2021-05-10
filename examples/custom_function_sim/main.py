import logging

import examples.basic.main as basic
import sim.docker as docker
from sim.core import Environment
from sim.faas import FunctionSimulator, FunctionReplica, FunctionRequest, SimulatorFactory, FunctionContainer
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


class CustomSimulatorFactory(SimulatorFactory):

    def __init__(self) -> None:
        super().__init__()

    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        return MyFunctionSimulator()


class MyFunctionSimulator(FunctionSimulator):

    def deploy(self, env: Environment, replica: FunctionReplica):
        # simulate a docker pull command for deploying the function (also done by sim.faassim.DockerDeploySimMixin)
        yield from docker.pull(env, replica.container.image, replica.node.ether_node)

    def startup(self, env: Environment, replica: FunctionReplica):
        logger.info('[simtime=%.2f] starting up function replica for function %s', env.now, replica.function.name)

        # you could create a very fine-grained setup routines here
        yield env.timeout(10)  # simulate docker startup

    def setup(self, env: Environment, replica: FunctionReplica):
        # no setup routine
        yield env.timeout(0)

    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        # you would probably either create one simulator per function, or use a generalized simulator, this is just
        # to demonstrate how the simulators are used to encapsulate simulator behavior.

        logger.info('[simtime=%.2f] invoking function %s on node %s', env.now, request, replica.node.name)

        # for full flexibility you decide the resources used
        cpu_millis = replica.node.capacity.cpu_millis * 0.1
        env.put_resource(replica, 'cpu', cpu_millis)
        node = replica.node

        node.current_requests.add(request)

        if replica.function.name == 'python-pi':
            if replica.node.name.startswith('rpi3'):  # those are nodes we created in basic.example_topology()
                yield env.timeout(20)  # invoking this function takes 20 seconds on a raspberry pi
            else:
                yield env.timeout(2)  # invoking this function takes 2 seconds on all other nodes in the cluster
        elif replica.function.name == 'resnet50-inference':
            yield env.timeout(0.5)  # invoking this function takes 500 ms
        else:
            yield env.timeout(0)

        # also, you have to release them at the end
        env.remove_resource(replica, 'cpu', cpu_millis)
        node.current_requests.remove(request)

    def teardown(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)


if __name__ == '__main__':
    main()
