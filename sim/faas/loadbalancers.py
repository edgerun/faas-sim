import abc
import logging
from typing import Generator, Optional, List, Any

import simpy
from faas.system import FunctionRequest, FunctionResponse, FunctionContainer
from faas.util.constant import function_label
from faas.util.rwlock import ReadWriteLock

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.faas import FunctionSimulator, SimFunctionReplica
from sim.faas.core import SimLoadBalancer

logger = logging.getLogger(__name__)


class LoadBalancerFunctionContainer(FunctionContainer):
    def __init__(self, fn_container: FunctionContainer):
        super(LoadBalancerFunctionContainer, self).__init__(fn_container.fn_image, fn_container.resource_config,
                                                            fn_container.labels)


class ForwardingClientFunctionContainer(FunctionContainer):
    """
    This class extends the regular FunctionContainer to include objects that are used to generate requests.
    """
    ia_generator: Generator
    size: int
    fn: SimFunctionDeployment
    lb_fn: SimFunctionDeployment
    # if not None, we consider this to be the maximum number of requests that should be generated
    # if None, it is considered to be the duration the client will generate requests
    max_requests: Optional[int]

    def __init__(self, fn_container: FunctionContainer, ia_generator: Generator,
                 size: int, fn: SimFunctionDeployment,
                 lb_fn: SimFunctionDeployment,
                 max_requests: Optional[int] = None):
        super(ForwardingClientFunctionContainer, self).__init__(fn_container.fn_image, fn_container.resource_config,
                                                                fn_container.labels)
        self.ia_generator = ia_generator
        self.lb_fn = lb_fn
        self.size = size
        self.fn = fn
        self.max_requests = max_requests


class ForwardingClientSimulator(FunctionSimulator):
    """
    This FunctionSimulator simulates a client that invokes a function.
    The advantage of that is, that the simulation will simulate any network traffic accurately.
    Which entails the function call between the client and the final destination (invoked function replica).
    """

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, Optional[FunctionResponse]]:
        try:
            container: ForwardingClientFunctionContainer = replica.container
            ia_generator = container.ia_generator
            max_requests = None
            if container.max_requests:
                max_requests = container.max_requests

            if max_requests is None:
                while True:
                    ia = next(ia_generator)
                    request = FunctionRequest(
                        container.lb_fn.name,
                        env.now,
                        client=replica.node.name,
                        size=container.size,
                        body=container.fn.name
                    )
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request)
            else:
                for _ in range(max_requests):
                    ia = next(ia_generator)
                    request = FunctionRequest(
                        container.lb_fn.name,
                        env.now,
                        client=replica.node.name,
                        size=container.size,
                        body=container.fn.name
                    )
                    yield env.timeout(ia)
                    yield from env.faas.invoke(request)
        except simpy.Interrupt:
            pass
        except StopIteration:
            logger.debug(f'{replica.function.name} gen has finished')
        except Exception as e:
            logger.error(e)
        finally:
            return None


class BaseLoadBalancerSimulator(FunctionSimulator, abc.ABC):
    """
    This FunctionSimulator acts as base for implementations Load Balancers that are scheduled as Pods.
    This allows for decentralized clients and load balancers and enables a full edge-cloud simulation.
    """

    def invoke(self, env: Environment, replica: SimFunctionReplica, request: FunctionRequest) -> Generator[
        None, None, Optional[FunctionResponse]]:
        next_replica = self.next_replica(env, replica, request)
        host = replica.node.name
        proxy_request = self._create_proxy_request(env, host, next_replica, request)
        response = yield from env.faas.invoke(proxy_request)
        # TODO might be interesting to insert here some headers (i.e., when the load balancer received the request,...)
        return response

    def next_replica(self, env: Environment, replica: SimFunctionReplica,
                     request: FunctionRequest) -> SimFunctionReplica:
        ...

    def _create_proxy_request(self, env: Environment, host: str, replica: SimFunctionReplica,
                              request: FunctionRequest) -> FunctionRequest:
        fn = request.body if replica.labels[function_label] == request.body else replica.function.name
        forwarded_true = {'lb-forwarded': True}
        if request.headers is not None:
            request.headers.update(forwarded_true)
        else:
            request.headers = forwarded_true
        return FunctionRequest(
            name=fn,
            start=env.now,
            size=request.size,
            request_id=request.request_id,
            body=request.body,
            client=host,
            replica=replica,
            headers=request.headers
        )


class LoadBalancerSimulator(BaseLoadBalancerSimulator):

    def __init__(self, lb: SimLoadBalancer):
        self.lb = lb

    def next_replica(self, env: Environment, replica: SimFunctionReplica,
                     request: FunctionRequest) -> SimFunctionReplica:
        modified_request = self.copy_request(replica.node.name, request)
        return self.lb.next_replica(modified_request)

    def copy_request(self, host: str, request: FunctionRequest) -> FunctionRequest:
        return FunctionRequest(
            request.body,
            start=request.start,
            size=request.size,
            request_id=request.request_id,
            body=request.body,
            client=host,
            headers=request.headers
        )

class UpdateableLoadBalancer(abc.ABC):

    def update(self, env: Environment):
        """
        This method is automatically called and should update the weights of this loadbalancer
        """
        ...

class LoadBalancerUpdateProcess():

    def __init__(self, reconcile_interval: int):
        self.load_balancers: List[UpdateableLoadBalancer] = []
        self.reconcile_interval = reconcile_interval
        self.rw_lock = ReadWriteLock()

    def run(self, env: Environment) -> Generator[simpy.events.Event, Any, Any]:
        for lb in self.load_balancers:
            lb.update(env)
        while True:
            yield env.timeout(self.reconcile_interval)
            logger.debug(f'Update load balancer weights')
            for lb in self.load_balancers:
                try:
                    lb.update(env)
                except Exception as e:
                    logger.error(e)

    def add(self, lb: UpdateableLoadBalancer):
        self.load_balancers.append(lb)
