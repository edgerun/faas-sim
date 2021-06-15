import logging
import random
from ext.jjnp21.load_balancers.localized_lrt import LocalizedLeastResponseTimeLoadBalancer
from ext.jjnp21.load_balancers.localized_rr import LocalizedRoundRobinLoadBalancer
from ext.jjnp21.load_balancers.lrt import LeastResponseTimeLoadBalancer
from sim.core import Environment, Node
from sim.faas import DefaultFaasSystem, FunctionRequest, FunctionState, LoadBalancer
from sim.faas.system import simulate_function_invocation
from ext.jjnp21.topology import get_client_nodes
from sim.net import SafeFlow

logger = logging.getLogger(__name__)


class LocalizedLoadBalancerFaasSystem(DefaultFaasSystem):
    def __init__(self, env: Environment, scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False) -> None:
        super().__init__(env, scale_by_requests, scale_by_average_requests, scale_by_queue_requests_per_replica)
        self.client_nodes = get_client_nodes(self.env.topology)

    def set_load_balancer(self, lb: LoadBalancer):
        self.load_balancer = lb

    def invoke(self, request: FunctionRequest):
        # TODO: how to return a FunctionResponse?
        logger.debug('invoking function %s', request.name)

        if request.name not in self.functions_deployments.keys():
            logger.warning('invoking non-existing function %s', request.name)
            return

        if len(self.client_nodes) > 0:
            request.client_node = random.choice(self.client_nodes)

        t_received = self.env.now

        replicas = self.get_replicas(request.name, FunctionState.RUNNING)
        if not replicas:
            '''
            https://docs.openfaas.com/architecture/autoscaling/#scaling-up-from-zero-replicas

            When scale_from_zero is enabled a cache is maintained in memory indicating the readiness of each function.
            If when a request is received a function is not ready, then the HTTP connection is blocked, the function is
            scaled to min replicas, and as soon as a replica is available the request is proxied through as per normal.
            You will see this process taking place in the logs of the gateway component.
            '''
            yield from self.poll_available_replica(request.name)

        if len(replicas) < 1:
            raise ValueError
        elif len(replicas) >= 1:
            logger.debug('asking load balancer for replica for request %s:%d', request.name, request.request_id)
            replica = self.next_replica(request)

        logger.debug('dispatching request %s:%d to %s', request.name, request.request_id, replica.node.name)

        lb_client_latency = 0
        lb_function_latency = 0
        tx_time = 0
        t_start = self.env.now

        if request.load_balancer is not None and hasattr(request, 'client_node'):
            start = self.env.now
        # right now I used 250kb request payload, which should be a small JPG with the added HTTP overhead
            yield from simulate_request_transfer(self.env, request.load_balancer.node.name, request.client_node.name, 250)
            tx_time = self.env.now - start

        if request.load_balancer is not None and hasattr(request, 'client_node'):
            latency = self.env.topology.latency(request.load_balancer.node, request.client_node) * 0.001
            lb_client_latency = latency
            yield self.env.timeout(latency * 2)

        if request.load_balancer is not None and \
                (isinstance(request.load_balancer, LocalizedLeastResponseTimeLoadBalancer)
                 or isinstance(request.load_balancer, LocalizedRoundRobinLoadBalancer)):
            lb_node = request.load_balancer.node
            lb_latency = self.env.topology.latency(lb_node, replica.node.ether_node) * 0.001
            lb_function_latency = lb_latency
            yield self.env.timeout(lb_latency * 2)  # * 2 for full round trip
        # else:
        #     print('lol')

        yield from simulate_function_invocation(self.env, replica, request)

        t_end = self.env.now
        if request.load_balancer is not None:
            if isinstance(request.load_balancer, LeastResponseTimeLoadBalancer):
                request.load_balancer.report_response_time(request, replica, t_end - t_start)
        # TODO Add load-balancer incurred routing delay here
        # TODO log info s.t. load-balancer can
        t_wait = t_start - t_received
        t_exec = t_end - t_start
        self.env.metrics.log_invocation(request.name, replica.image, replica.node.name, t_wait, t_start,
                                        t_exec, id(replica), lb_client_latency=lb_client_latency,
                                        lb_function_latency=lb_function_latency, tx_time=tx_time)


def simulate_request_transfer(env: Environment, src_name: str, dest_name: str, size_kb: int) -> float:
    route = env.topology.route_by_node_name(src_name, dest_name)
    flow = SafeFlow(env, size_kb * 1024, route)
    start = env.now
    yield flow.start()
    end = env.now
    # return end - start