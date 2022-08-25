from faas.context import PlatformContext, NodeService, ZoneService

from sim.context.platform.deployment.service import SimFunctionDeploymentService
from sim.context.platform.network.service import SimNetworkService
from sim.context.platform.node.model import SimFunctionNode
from sim.context.platform.replica.service import SimFunctionReplicaService
from sim.context.platform.telemetry.service import SimTelemetryService
from sim.context.platform.trace.service import SimTraceService


class SimPlatformContext(PlatformContext):
    deployment_service: SimFunctionDeploymentService
    network_service: SimNetworkService
    node_service: NodeService[SimFunctionNode]
    replica_service: SimFunctionReplicaService
    telemetry_service: SimTelemetryService
    trace_service: SimTraceService
    zone_service: ZoneService
