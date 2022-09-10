import logging
from abc import ABC

import simpy
from faas.system.core import FunctionRequest

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment

__all__ = [
    'function_trigger',
    'FunctionRequestFactory',
    'SimpleFunctionRequestFactory'
]


class FunctionRequestFactory(ABC):
    def generate(self, env: Environment, deployment: SimFunctionDeployment) -> FunctionRequest: ...


class SimpleFunctionRequestFactory(FunctionRequestFactory):

    def __init__(self, client: str = None, size: float = None):
        self.client = client
        self.size = size

    def generate(self, env: Environment, deployment: SimFunctionDeployment) -> FunctionRequest:
        now = env.now
        return FunctionRequest(deployment.name, now, client=self.client, size=self.size)


def function_trigger(env: Environment, deployment: SimFunctionDeployment, fn_request_factory: FunctionRequestFactory,
                     ia_generator, max_requests=None):
    try:
        if max_requests is None:
            while True:
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.process(env.faas.invoke(fn_request_factory.generate(env, deployment)))
        else:
            for _ in range(max_requests):
                ia = next(ia_generator)
                yield env.timeout(ia)
                env.process(env.faas.invoke(fn_request_factory.generate(env, deployment)))

    except simpy.Interrupt:
        pass
    except StopIteration:
        logging.error(f'{deployment.name} gen has finished')
