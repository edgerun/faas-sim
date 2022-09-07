from sim.context.model import SimPlatformContext
from sim.context.platform.deployment.factory import create_deployment_service
from sim.context.platform.network.factory import create_network_service
from sim.context.platform.node.factory import create_node_service
from sim.context.platform.replica.factory import create_replica_service
from sim.context.platform.request.factory import create_request_service
from sim.context.platform.telemetry.factory import create_telemetry_service
from sim.context.platform.trace.factory import create_trace_service
from sim.context.platform.zone.factory import create_zone_service
from sim.core import Environment


def create_platform_context(env: Environment) -> SimPlatformContext:
    deployment_service = create_deployment_service(env.metrics)

    network_service = create_network_service(env.topology)

    node_service = create_node_service(env, env.topology)

    replica_service = create_replica_service(node_service, deployment_service, env)

    # TODO let users inject windowsize for telemetry service
    telemetry_service = create_telemetry_service(120, env)

    # TODO let users inject windowsize for trace service
    trace_service = create_trace_service(60, node_service, lambda: env.now)

    zone_service = create_zone_service(node_service.get_zones())

    request_service = create_request_service(env)

    context = SimPlatformContext(
        deployment_service=deployment_service,
        network_service=network_service,
        node_service=node_service,
        replica_service=replica_service,
        telemetry_service=telemetry_service,
        trace_service=trace_service,
        zone_service=zone_service
    )
    context.request_service = request_service
    return context
