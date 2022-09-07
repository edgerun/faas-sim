from typing import Optional, Callable

from faas.context import ResponseRepresentation, NodeService, InMemoryTraceService
from faas.system import FunctionResponse

from sim.context.platform.node.model import SimFunctionNode
from sim.context.platform.trace.service import SimTraceService


def create_trace_service(window_size: int, node_service: NodeService[SimFunctionNode], now: Callable[[], float]):
    parser = create_parse_request_factory(node_service)
    return SimTraceService(now, window_size, node_service, parser)


def create_parse_request_factory(node_service: NodeService[SimFunctionNode]):
    def parse_request(response: FunctionResponse) -> Optional[ResponseRepresentation]:
        sent = response.request.start
        done = response.ts_end
        rtt = done - sent
        if response.request.client is not None:
            client_cluster = node_service.find(response.request.client).cluster
        else:
            client_cluster = 'N/A'
        dest_cluster = response.replica.node.cluster
        return ResponseRepresentation(
            ts=done,
            function=response.replica.function.name,
            function_image=response.replica.container.fn_image.image,
            replica_id=response.replica.replica_id,
            node=response.replica.node.name,
            rtt=rtt,
            done=done,
            sent=sent,
            origin_zone=client_cluster,
            dest_zone=dest_cluster,
            client=response.client,
            status=response.code,
            request_id=response.request_id
        )

    return parse_request
