import logging
from dataclasses import dataclass
from typing import Generator

import simpy

from .core import FunctionSimulator, FunctionRequest, SimFunctionReplica, FunctionSimulatorResponse
from ..core import Environment

logger = logging.getLogger(__name__)


@dataclass
class WatchdogResponse:
    body: str
    code: int
    size: int


class Watchdog(FunctionSimulator):

    def claim_resources(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest): ...

    def release_resources(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest): ...

    def execute(self, env: Environment, replica: SimFunctionReplica,
                request: FunctionRequest) -> Generator[None, None, WatchdogResponse]: ...


class ForkingWatchdog(Watchdog):

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, FunctionSimulatorResponse]:
        replica.node.current_requests.add(request)
        t_fet_start = env.now

        logger.debug('[simtime=%.2f] invoking function %s on node %s', t_fet_start, request, replica.node.name)

        yield from self.claim_resources(env, replica, request)

        response: WatchdogResponse = yield from self.execute(env, replica, request)

        yield from self.release_resources(env, replica, request)

        t_fet_end = env.now
        fet = t_fet_end - t_fet_start
        env.metrics.log_fet(replica, request, t_fet_start=t_fet_start, t_fet_end=t_fet_end)

        replica.node.current_requests.remove(request)
        return FunctionSimulatorResponse(
            body=response.body,
            size=response.size,
            code=response.code,
            t_wait=t_fet_start,
            t_exec=t_fet_start,
            fet=fet
        )


class HTTPWatchdog(Watchdog):
    queue: simpy.Resource

    def __init__(self, workers: int):
        self.workers = workers
        self.queue = None

    def setup(self, env: Environment, replica: SimFunctionReplica):
        self.queue = simpy.Resource(env, capacity=self.workers)

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, FunctionSimulatorResponse]:
        token = self.queue.request()
        t_wait_start = env.now
        yield token

        t_fet_start = env.now
        logger.debug('[simtime=%.2f] invoking function %s on node %s', t_fet_start, request, replica.node.name)

        replica.node.current_requests.add(request)

        yield from self.claim_resources(env, replica, request)

        response: WatchdogResponse = yield from self.execute(env, replica, request)

        yield from self.release_resources(env, replica, request)

        t_fet_end = env.now

        replica.node.current_requests.remove(request)
        fet = t_fet_end - t_fet_start
        env.metrics.log_fet(replica, request, t_fet_start=t_fet_start, t_fet_end=t_fet_end)

        self.queue.release(token)

        return FunctionSimulatorResponse(
            body=response.body,
            size=response.size,
            code=response.code,
            t_wait=t_wait_start,
            t_exec=t_fet_start,
            fet=fet
        )
