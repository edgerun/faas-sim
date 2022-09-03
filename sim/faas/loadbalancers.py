import abc
import logging
import random
from collections import Counter, defaultdict
from typing import Generator, Optional, Any, Dict
from typing import List

import simpy
from faas.system import FunctionRequest, FunctionResponse, FunctionContainer
from faas.system.loadbalancer import LoadBalancer
from faas.util.constant import function_label
from faas.util.rwlock import ReadWriteLock

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.faas import FunctionSimulator, SimFunctionReplica, SimLoadBalancer
from sim.faas import LocalizedSimLoadBalancer

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


class UpdateableLoadBalancer(SimLoadBalancer):

    def update(self, weights: Dict[str, Dict[str, float]]):
        """
        This method is automatically called and should update the weights of this loadbalancer accordingly
        """
        ...


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


class UpdateableLoadBalancerSimulator(BaseLoadBalancerSimulator):

    def __init__(self, lb: UpdateableLoadBalancer):
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


class LoadBalancerUpdateProcess():

    def __init__(self, reconcile_interval: int):
        self.load_balancers: List[LoadBalancer] = []
        self.reconcile_interval = reconcile_interval
        self.rw_lock = ReadWriteLock()

    def run(self, env: Environment) -> Generator[simpy.events.Event, Any, Any]:
        for lb in self.load_balancers:
            lb.update()
        while True:
            yield env.timeout(self.reconcile_interval)
            logger.debug(f'Update load balancer weights')
            for lb in self.load_balancers:
                try:
                    lb.update()
                except Exception as e:
                    logger.error(e)

    def add(self, lb: LoadBalancer):
        self.load_balancers.append(lb)


"""
Basically all of this code stems from the jacob-thesis branch.
Thanks, @jjnp for this implementation.
"""


class WRRProvider(abc.ABC):
    replica_ids: List[str]

    @abc.abstractmethod
    def next_id(self) -> str:
        pass


class SmoothWeightedRoundRobinProvider(WRRProvider):
    def __init__(self, weights: Dict[str, float], scaling: float = 2.5):
        self.current_values: Dict[str, float] = defaultdict(lambda: 0)
        self.scaling = scaling
        if len(weights) > 0:
            self.max_weight = max(weights.values())
            self.weight_sum = sum(weights.values())
        else:
            self.max_weight = 0
            self.weight_sum = 0
        self.weights = weights
        self.replica_ids = list(weights.keys())
        if len(weights) > 0:
            for replica_id in weights.keys():
                self.current_values[replica_id] = 0

    def next_id(self) -> str:
        for replica_id, weight in self.weights.items():
            self.current_values[replica_id] += weight
        chosen_id = max(self.current_values, key=self.current_values.get)
        self.current_values[chosen_id] -= self.weight_sum
        return chosen_id


class DefaultWRRProvider(WRRProvider):
    def __init__(self, weights: Dict[str, float], scaling: float = 1.0):
        self.gcd = 1
        self.scaling = scaling
        self.replica_ids = list(weights.keys())
        random.shuffle(self.replica_ids)
        self.weights = dict()
        self.cw = 0
        self.last = -1
        self.n = len(weights)
        self.max_weight = 1
        self.weights = weights

    def __str__(self):
        return str(self.weights)

    def _calculate_gcd(self) -> int:
        weights = self.weights.values()
        max_gcd = min(weights)
        gcd = 1
        for i in range(max_gcd, 0, -1):
            valid = True
            for w in weights:
                if w % i != 0:
                    valid = False
                    break
            if valid and i > 1:
                gcd = i
                break
        # print(f'GCD calclated is: {gcd}')
        return gcd

    def next_id(self) -> str:
        while True:
            self.last = (self.last + 1) % self.n
            if self.last == 0:
                self.cw -= self.gcd
                if self.cw <= 0:
                    self.cw = self.max_weight
            if self.weights[self.replica_ids[self.last]] >= self.cw:
                return self.replica_ids[self.last]


class WrrProviderFactory():

    def create(self, weights: Dict[str, float]) -> WRRProvider: ...


class SmoothWeightedRoundRobinProviderFactory(WrrProviderFactory):

    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling

    def create(self, weights: Dict[str, float]) -> WRRProvider:
        return SmoothWeightedRoundRobinProvider(weights, self.scaling)


class DefaultWrrProviderFactory(WrrProviderFactory):

    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling

    def create(self, weights: Dict[str, float]) -> WRRProvider:
        return DefaultWRRProvider(weights, self.scaling)


class WrrLoadBalancer(UpdateableLoadBalancer, LocalizedSimLoadBalancer):
    # TODOs
    # [x] Check for new functions and new replicas + integrate them
    # [x] Integrate new replicas in a smarter way (currently we just reset the metrics provider)
    # [x] pay attention to node state (running, etc.)

    def __init__(self, env: Environment, cluster: str, wrr_factory: WrrProviderFactory) -> None:
        super().__init__(env, cluster)
        self.count = Counter()
        self.wrr_factory = wrr_factory
        self.wrr_providers: Dict[str, WRRProvider] = dict()

    def next_replica(self, request: FunctionRequest) -> Optional[SimFunctionReplica]:
        managed_replicas = self.get_running_replicas(request.name)
        if len(managed_replicas) == 0:
            return None
        return self._replica_by_id(request.name, self.wrr_providers[request.name].next_id())

    def update(self, weights: Dict[str, Dict[str, float]]):
        managed_functions = self.get_functions()
        for function in managed_functions:
            function_name = function.name
            self.wrr_providers[function_name] = self.wrr_factory.create(weights[function_name])

    def _replica_by_id(self, function: str, replica_id: str) -> Optional[SimFunctionReplica]:
        for r in self.get_running_replicas(function):
            if r.replica_id == replica_id:
                return r
        return None
