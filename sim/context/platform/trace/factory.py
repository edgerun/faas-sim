from typing import Optional, Callable

from faas.context import ResponseRepresentation, NodeService, InMemoryTraceService
from faas.system import FunctionResponse

from sim.context.platform.node.model import SimFunctionNode
from sim.context.platform.trace.service import SimTraceService
from sim.context.platform.replica.service import SimFunctionReplicaService


def create_trace_service(window_size: int, node_service: NodeService[SimFunctionNode], replica_service: SimFunctionReplicaService, now: Callable[[], float]):
    parser = create_parse_request_factory(replica_service)
    return SimTraceService(now, window_size, node_service, parser)


def create_parse_request_factory(replica_service: SimFunctionReplicaService):
    def parse_request(response: FunctionResponse) -> Optional[ResponseRepresentation]:
        sent = response.request.start
        done = response.ts_end
        rtt = done - sent
        if response.request.client is not None:
            replica_id = response.request.client
            replica = replica_service.get_function_replica_by_id(replica_id)
            client_cluster = replica.node.cluster
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
