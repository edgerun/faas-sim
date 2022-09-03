import logging
from typing import Generator, Optional

import simpy
from faas.system import FunctionRequest, FunctionResponse, FunctionContainer

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.faas import FunctionSimulator, SimFunctionReplica
from sim.requestgen import FunctionRequestFactory, SimpleFunctionRequestFactory

logger = logging.getLogger(__name__)


class ClientSimulator(FunctionSimulator):
    """
    This FunctionSimulator simulates a client that invokes a function.
    The advantage of that is, that the simulation will simulate any network traffic accurately.
    Which entails the function call between the client and the final destination (invoked function replica).
    """

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, Optional[FunctionResponse]]:
        container: ClientFunctionContainer = replica.container
        request_factory: SimpleFunctionRequestFactory = container.fn_request_factory
        request_factory.client = replica.node.name

        # read generator parameters
        fn_deployment = replica.container.fn
        try:
            container: ClientFunctionContainer = replica.container
            ia_generator = container.ia_generator
            max_requests = None
            if container.max_requests:
                max_requests = container.max_requests

            if max_requests is None:
                while True:
                    ia = next(ia_generator)
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request_factory.generate(env, fn_deployment))
            else:
                for _ in range(max_requests):
                    ia = next(ia_generator)
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request_factory.generate(env, fn_deployment))

        except simpy.Interrupt:
            pass
        except StopIteration:
            logger.debug(f'{replica.function.name} gen has finished')
        finally:
            # return FunctionResponse(request, request.request_id, request.client, request.name, request.body, 200, None,
            #                         None, None, None)
            return None

class ClientFunctionContainer(FunctionContainer):
    """
    This class extends the regular FunctionContainer to include objects that are used to generate requests.
    """
    ia_generator: Generator
    fn_request_factory: FunctionRequestFactory
    fn: SimFunctionDeployment
    # if True, we consider this to be the maximum number of requests that should be generated
    # if False, it is considered to be the duration the client will generate requests
    max_requests: Optional[int]

    def __init__(self, fn_container: FunctionContainer, ia_generator: Generator,
                 fn_request_factory: FunctionRequestFactory, fn: SimFunctionDeployment,
                 max_requests: Optional[int] = None):
        super(ClientFunctionContainer, self).__init__(fn_container.fn_image, fn_container.resource_config,
                                                      fn_container.labels)
        self.ia_generator = ia_generator
        self.fn_request_factory = fn_request_factory
        self.fn = fn
        self.max_requests = max_requests


