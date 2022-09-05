import abc
import logging

import simpy
from ether.core import Flow, Route, UninterruptingFlow

from sim.net import SafeFlow

logger = logging.getLogger(__name__)


class FlowFactory(abc.ABC):

    def create_flow(self, env: simpy.Environment, size: int, route: Route) -> Flow: ...


class UninterruptingFlowFactory(FlowFactory):
    def create_flow(self, env: simpy.Environment, size: int, route: Route) -> Flow:
        return UninterruptingFlow(env, size, route)


class SafeFlowFactory(FlowFactory):
    def create_flow(self, env: simpy.Environment, size: int, route: Route) -> Flow:
        return SafeFlow(env, size, route)
