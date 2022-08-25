import time
from typing import Optional, Any, Generator, Callable, List, Dict

import simpy
from sklearn.base import RegressorMixin


class SimulationTimeoutError(BaseException):
    pass


class Environment(simpy.Environment):
    cluster: 'SimulationClusterContext'
    faas: 'FaasSystem'

    def __init__(self, initial_time=0):
        super().__init__(initial_time)
        self.faas = None
        self.simulator_factory = None
        self.topology = None
        self.storage_index = None
        self.benchmark = None
        self.cluster = None
        self.container_registry = None
        self.metrics = None
        self.scheduler = None
        self.resource_state = None
        self.resource_monitor = None
        self.flow_factory = None
        self.context = None
        self.background_processes: List[Callable[[Environment], Generator[simpy.events.Event, Any, Any]]] = []
        self.degradation_models: Dict[str, Optional[RegressorMixin]] = {}


def timeout_listener(env, started, max_time, interval=1):
    while True:
        yield env.timeout(interval)

        if (time.time() - started) > max_time:
            raise SimulationTimeoutError()
