import logging
from dataclasses import dataclass
from typing import Generator, List

import simpy

from .core import FunctionSimulator, FunctionRequest, SimFunctionReplica, FunctionSimulatorResponse
from ..context.platform.request.service import RequestService
from ..core import Environment

logger = logging.getLogger(__name__)


@dataclass
class WatchdogResponse:
    body: str
    code: int
    size: int


class Watchdog(FunctionSimulator):

    def __init__(self, teardown_wait_interval: int = 5):
        self.teardown_wait_interval = teardown_wait_interval

    def claim_resources(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        simpy.Event, None, List[int]]: ...

    def release_resources(self, env: Environment, replica: SimFunctionReplica, resource_indices: List[int]): ...

    def execute(self, env: Environment, replica: SimFunctionReplica,
                request: FunctionRequest) -> Generator[None, None, WatchdogResponse]: ...

    def teardown(self, env: Environment, replica: SimFunctionReplica):
        request_service: RequestService = env.context.request_service
        while len(request_service.get_inflight_requests_of_replica(replica)) != 0:
            yield env.timeout(self.teardown_wait_interval)
        yield env.timeout(0)


class ForkingWatchdog(Watchdog):

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, FunctionSimulatorResponse]:
        env.context.request_service.add_request(request)
        ts_fet_start = env.now

        logger.debug('[simtime=%.2f] invoking function %s on node %s', ts_fet_start, request, replica.node.name)

        resource_indices = yield from self.claim_resources(env, replica, request)

        response: WatchdogResponse = yield from self.execute(env, replica, request)

        yield from self.release_resources(env, replica, resource_indices)

        ts_fet_end = env.now
        fet = ts_fet_end - ts_fet_start
        env.metrics.log_fet(replica, request, ts_fet_start=ts_fet_start, ts_fet_end=ts_fet_end)

        env.context.request_service.remove_request(request.request_id)
        return FunctionSimulatorResponse(
            body=response.body,
            size=response.size,
            code=response.code,
            ts_wait=ts_fet_start,
            ts_exec=ts_fet_start,
            fet=fet
        )


class HTTPWatchdog(Watchdog):
    queue: simpy.Resource

    def __init__(self, workers: int, teardown_wait_interval: int = 5):
        super(HTTPWatchdog, self).__init__(teardown_wait_interval)
        self.workers = workers
        self.queue = None

    def setup(self, env: Environment, replica: SimFunctionReplica):
        self.queue = simpy.Resource(env, capacity=self.workers)

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, FunctionSimulatorResponse]:
        token = self.queue.request()
        ts_wait_start = env.now
        yield token

        ts_fet_start = env.now
        logger.debug('[simtime=%.2f] invoking function %s on node %s', ts_fet_start, request, replica.node.name)

        env.context.request_service.add_request(request)

        resource_indices = yield from self.claim_resources(env, replica, request)

        response: WatchdogResponse = yield from self.execute(env, replica, request)

        yield from self.release_resources(env, replica, resource_indices)

        ts_fet_end = env.now

        env.context.request_service.remove_request(request)
        fet = ts_fet_end - ts_fet_start
        env.metrics.log_fet(replica, request, ts_fet_start=ts_fet_start, ts_fet_end=ts_fet_end,
                            ts_wait_start=ts_wait_start)

        self.queue.release(token)

        return FunctionSimulatorResponse(
            body=response.body,
            size=response.size,
            code=response.code,
            ts_wait=ts_wait_start,
            ts_exec=ts_fet_start,
            fet=fet
        )
