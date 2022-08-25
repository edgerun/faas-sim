from typing import Optional

from faas.context import ResponseRepresentation, NodeService, InMemoryTraceService
from faas.system import FunctionResponse

from sim.context.platform.node.model import SimFunctionNode
from sim.context.platform.trace.service import SimTraceService


def create_trace_service(window_size: int, node_service: NodeService[SimFunctionNode]):
    parser = create_parse_request_factory(node_service)
    trace_service = InMemoryTraceService[FunctionResponse](window_size, node_service, parser)
    return SimTraceService(trace_service)


def create_parse_request_factory(node_service: NodeService[SimFunctionNode]):
    def parse_request(response: FunctionResponse) -> Optional[ResponseRepresentation]:
        sent = response.request.start
        done = response.end
        rtt = done - sent
        client_cluster = node_service.find(response.request.client).cluster
        dest_cluster = node_service.find(response.replica.node.cluster).cluster
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
        )

    return parse_request
