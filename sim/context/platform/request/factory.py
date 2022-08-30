from sim.context.platform.request.service import SimpleRequestService, RequestService
from sim.core import Environment


def create_request_service(env: Environment) -> RequestService:
    return SimpleRequestService(lambda: env.now)
