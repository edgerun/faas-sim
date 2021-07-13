import logging
import random
from enum import Enum

from ext.jjnp21.ether_customization.custom_ether import UninterruptingFlow
from ext.jjnp21.load_balancers.localized_lrt import LocalizedLeastResponseTimeLoadBalancer
from ext.jjnp21.load_balancers.localized_rr import LocalizedRoundRobinLoadBalancer
from ext.jjnp21.load_balancers.lrt import LeastResponseTimeLoadBalancer
from sim.core import Environment, Node
from sim.faas import DefaultFaasSystem, FunctionRequest, FunctionState, LoadBalancer, FunctionReplica
from sim.faas.system import simulate_function_invocation
from ext.jjnp21.topology import get_client_nodes
from sim.net import SafeFlow

logger = logging.getLogger(__name__)

class NetworkSimulationMode(Enum):
    FAST = 1
    ACCURATE = 2


class LocalizedLoadBalancerFaasSystem(DefaultFaasSystem):
    def __init__(self, env: Environment, scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False,
                 scale_static: bool = False, net_mode: NetworkSimulationMode = NetworkSimulationMode.ACCURATE) -> None:
        super().__init__(env, scale_by_requests, scale_by_average_requests, scale_by_queue_requests_per_replica,
                         scale_static=scale_static)
        self.client_nodes = get_client_nodes(self.env.topology)
        self.net_mode = net_mode

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

        if replica is None:
            # This can occur very infrequently when a replica gets removed while a request is in-flight for that replica.
            # It is rare enough to simply ignore (once every 30-50k reuests if there's hundreds of replicas)
            # We just need to address it to make sure we don't crash long experiments
            logger.warning('got a replica with value None. Ignoring and continuing on')
            return
        logger.debug('dispatching request %s:%d to %s', request.name, request.request_id, replica.node.name)

        t_start = self.env.now

        yield from self.simulate_function_invocation(replica, request)

        t_end = self.env.now


        t_wait = t_start - t_received
        t_exec = t_end - t_start
        client_city = 'N/A'
        client_node = 'N/A'
        lb_node = 'N/A'
        tx_time_cl_lb = 0
        tx_time_lb_fx = 0
        if hasattr(request, 'load_balancer') and request.load_balancer is not None:
            lb_node = request.load_balancer.node.name
        if hasattr(request, 'client_node') and request.client_node is not None:
            client_node = request.client_node.name
        if hasattr(request, 'tx_time_cl_lb'):
            tx_time_cl_lb = request.tx_time_cl_lb
        if hasattr(request, 'tx_time_lb_fx'):
            tx_time_lb_fx = request.tx_time_lb_fx
        if hasattr(request, 'client_node'):
            client_city = request.client_node.labels.get('city')

        self.env.metrics.log_invocation(request.name, replica.image, replica.node.name, t_wait, t_start,
                                        t_exec, id(replica),
                                        tx_time_cl_lb=tx_time_cl_lb,
                                        tx_time_lb_fx=tx_time_lb_fx,
                                        replica_city=replica.node.ether_node.labels.get('city'),
                                        client_city=client_city,
                                        client_node=client_node,
                                        lb_node=lb_node)

    def simulate_function_invocation(self, replica: FunctionReplica, request: FunctionRequest) -> (float, float):
        """
        Adapted version of "simulate function invocation" that also includes network simulation
        @param replica: The function replica
        @param request: The request to be processed
        @return: A tuple of two floats denoting
        (transfer time client<->load balancer, transfer time load balancer<->function replica)
        or None if there's no load balancer or client to be found
        """
        tx_time_cl_lb = 0
        tx_time_lb_fx = 0
        t_start = self.env.now

        if request.load_balancer is not None and hasattr(request, 'client_node'):
            # right now I used 250kb request payload, which should be a small JPG with the added HTTP overhead
            cl_lb_start = self.env.now
            yield from self.simulate_request_transfer(request.load_balancer.node.name, request.client_node.name, 250)
            tx_time_cl_lb = self.env.now - cl_lb_start
            lb_fx_start = self.env.now
            yield from self.simulate_request_transfer(request.load_balancer.node.name, replica.node.ether_node.name,
                                                      250)
            tx_time_lb_fx = self.env.now - lb_fx_start

        # actual function simulation portion
        yield from simulate_function_invocation(self.env, replica, request)
        t_end = self.env.now

        # report the total response time to the load-balancer for weight updates etc.
        if request.load_balancer is not None:
            if isinstance(request.load_balancer, LeastResponseTimeLoadBalancer):
                request.load_balancer.report_response_time(request, replica, t_end - t_start)

        if tx_time_cl_lb != 0 and tx_time_lb_fx != 0:
            request.tx_time_cl_lb = tx_time_cl_lb
            request.tx_time_lb_fx = tx_time_lb_fx

    def simulate_request_transfer(self, src_name: str, dest_name: str, size_kb: int) -> float:
        route = self.env.topology.route_by_node_name(src_name, dest_name)
        if len(route.hops) == 0:
            return
        flow = None
        if self.net_mode == NetworkSimulationMode.ACCURATE:
            flow = SafeFlow(self.env, size_kb * 1024, route)
        elif self.net_mode == NetworkSimulationMode.FAST:
            flow = UninterruptingFlow(self.env, size_kb * 1024, route)
        yield flow.start()



