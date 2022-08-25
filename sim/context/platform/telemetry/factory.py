from sim.context.platform.telemetry.service import SimTelemetryService
from sim.core import Environment


def create_telemetry_service(window: int, env: Environment):
    return SimTelemetryService(window, env)
