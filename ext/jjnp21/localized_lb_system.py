import itertools
import itertools
import logging
import random
from collections import defaultdict, Counter
from enum import Enum
from typing import List, Dict

import simpy

from ext.jjnp21.automator.factories.lb_scaler import LoadBalancerScalerFactory
from ext.jjnp21.core import LoadBalancerDeployment, LoadBalancerReplica
from ext.jjnp21.ether_customization.custom_ether import UninterruptingFlow
from ext.jjnp21.load_balancers.lrt import LeastResponseTimeLoadBalancer
from ext.jjnp21.scalers.lb_scaler import LoadBalancerScaler
from ext.jjnp21.topology import get_client_nodes
from sim.core import Environment
from sim.faas import DefaultFaasSystem, FunctionRequest, FunctionState, LoadBalancer, FunctionReplica, \
    FunctionDeployment, FunctionContainer
from sim.faas.system import simulate_function_invocation
from sim.net import SafeFlow

logger = logging.getLogger(__name__)


class NetworkSimulationMode(Enum):
    FAST = 1
    ACCURATE = 2


class LoadBalancerCapableFaasSystem(DefaultFaasSystem):

    def __init__(self, env: Environment, lb_scaler_factory: LoadBalancerScalerFactory,
                 scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False,
                 scale_static: bool = False):
        super().__init__(env, scale_by_requests, scale_by_average_requests, scale_by_queue_requests_per_replica,
                         scale_static=scale_static)
        self.lb_deployments: Dict[str, LoadBalancerDeployment] = {}
        # todo get LB scaler factory
        self.lb_scaler_factory: LoadBalancerScalerFactory = lb_scaler_factory
        self.lb_scalers: Dict[str, LoadBalancerScaler] = {}
        self.lb_replicas: Dict[str, List[LoadBalancerReplica]] = defaultdict(list)
        self.lb_scheduler_queue = simpy.Store(env)
        self.lb_replica_count = Counter()
        self.lb_replica_per_image_count = Counter()

    def poll_available_faas_replica(self, fn: str = None, interval=0.5):
        if fn is not None:
            while not self.get_lb_replicas(fn, FunctionState.RUNNING):
                yield self.env.timeout(interval)
        else:
            while not self.get_all_lb_replicas(state=FunctionState.RUNNING):
                yield self.env.timeout(interval)

    def get_load_balancer(self, request: FunctionRequest) -> LoadBalancer:
        """
        Returns a load-balancer instance to be used for handling the passed request
        @param request: the request for which you want to get a lb-instance
        @return: a valid LB instance for the request
        """
        # this is currently simply a random choice between all replicas
        all_replicas = self.get_all_lb_replicas(state=FunctionState.RUNNING)
        return random.choice(all_replicas).load_balancer

    def get_all_lb_replicas(self, state: FunctionState = None) -> List[LoadBalancerReplica]:
        """
        returns all load-balancer replicas, optionally filtered to match a statte
        @param state: optional. the state the returned replicas should have
        @return: a list of lb replicas
        """
        if state is None:
            return list(itertools.chain.from_iterable(self.lb_replicas.values()))
        return list(filter(lambda r: r.state == state, itertools.chain.from_iterable(self.lb_replicas.values())))

    def get_lb_replicas(self, lb_name: str, state=None) -> List[LoadBalancerReplica]:
        """
        Returns replicas for a given function, optionally filtered by state
        @param lb_name: the load-balancer name for which a replica should be retrieved
        @param state: optional. The state by which should be filtered (e.g. only running replicas)
        @return: A list of load balancer replicas, optionally matching the passed state
        """
        if state is None:
            return self.lb_replicas[lb_name]
        return [r for r in self.lb_replicas[lb_name] if r.state == state]

    def get_lb_deployments(self) -> List[FunctionDeployment]:
        return list(self.lb_deployments.values())

    def deploy_lb(self, ld: LoadBalancerDeployment):
        if ld.name in self.lb_deployments:
            raise ValueError('LB function already present')
        self.lb_deployments[ld.name] = ld
        # set up scaler
        scaler = self.lb_scaler_factory.create(ld, self.env)
        self.lb_scalers[ld.name] = scaler
        self.env.process(self.lb_scalers[ld.name].run())
        # not using yet, since I don't know what it does at all.
        # for f in ld.fn_containers:
        # self.function_containers[f.image] = f
        self.env.metrics.log_function_deployment(ld)
        self.env.metrics.log_function_deployment_lifecycle(ld, 'deploy')
        logger.info('deploying function %s with scale_min=%d', ld.name, ld.scaling_config.scale_min)
        yield from self.scale_up_lb(ld.name, ld.scaling_config.scale_min)

    def deploy_lb_replica(self, fd: FunctionDeployment, fn: FunctionContainer):
        replica = self.create_lb_replica(fd, fn)
        self.lb_replicas[fd.name].append(replica)
        self.env.metrics.log_queue_schedule(replica)
        self.env.metrics.log_function_replica(replica)
        yield self.lb_scheduler_queue.put(replica)

    def scale_up_lb(self, lb_name: str, add_count: int):
        ld = self.lb_deployments[lb_name]
        scaling_config = ld.scaling_config
        ranking = ld.ranking
        corrected_add_count = add_count

        if self.lb_replica_count.get(lb_name, None) is None:
            self.lb_replica_count[lb_name] = 0
        if self.replica_count[lb_name] >= scaling_config.scale_max:
            logger.debug('Load balancer %s wanted to scale up, but maximum number of replicas reached', lb_name)
            return
        if self.replica_count[lb_name] + add_count > scaling_config.scale_max:
            corrected_add_count = scaling_config.scale_max - self.replica_count[lb_name]
            logger.debug('Load balancer %s wanted to scale by %d replicas. To not exceed the configured maximum it will'
                         'only scale up by %d replicas instead.' % (lb_name, add_count, corrected_add_count))

        added_replica_count = 0
        for index, service in enumerate(ld.get_services()):
            # the different "services" here are all the same function just different images for the different
            # architectures and accelerators, e.g. TPU, GPU, x86, aarch64, etc.
            remaining_add_count = corrected_add_count - added_replica_count
            # If we added the required number of replicas, return
            if remaining_add_count + added_replica_count > corrected_add_count:
                print('test-statement')  # todo remove this if block. If all works correctly it should be unreachable
                return
            max_allowed_replicas = int(ranking.function_factor[service.image] * scaling_config.scale_max)
            # Check if this would spawn more instances than allowed for that image according to the function_factor
            if max_allowed_replicas < remaining_add_count + self.lb_replica_per_image_count[service.image]:
                remaining_add_count = max_allowed_replicas - self.lb_replica_per_image_count[service.image]
            for _ in range(remaining_add_count):
                yield from self.deploy_replica(ld, ld.get_container(service.image), ld.get_containers()[index:])
                added_replica_count += 1

    def scale_down_lb(self, lb_name: str, remove_count: int):
        current_replica_count = len(self.get_lb_replicas(lb_name, state=FunctionState.RUNNING))
        scale_min = self.lb_deployments[lb_name].scaling_config.scale_min
        # if remove_count > current_replica_count -> remove_count = current_replica_count
        if current_replica_count - remove_count < scale_min:
            remove_count = current_replica_count - scale_min
        logger.info(f'scale down {lb_name} by {remove_count}')
        self.env.metrics.log_scaling(lb_name, -remove_count)
        replicas_to_remove = self.choose_lb_replicas_to_remove(lb_name, remove_count)
        for r in replicas_to_remove:
            self.remove_lb_replica(r)

    def choose_lb_replicas_to_remove(self, lb_name: str, cnt: int):
        # currently the most recently added ones are being removed. This will be replaced with proper implementations
        running_replicas = self.get_lb_replicas(lb_name, FunctionState.RUNNING)
        return running_replicas[len(running_replicas) - cnt:]

    def create_lb_replica(self, ld: LoadBalancerDeployment, fn: FunctionContainer) -> LoadBalancerReplica:
        replica = LoadBalancerReplica()
        # todo use the LB specific stuff here. Not added yet.
        replica.function = ld
        replica.container = fn
        replica.load_balancer = ld.create_load_balancer(self.env)
        replica.pod = self.create_pod(ld, fn)
        return replica

    def remove_lb_replica(self, replica: LoadBalancerReplica):
        node = replica.node.skippy_node
        self.env.metrics.log_teardown(replica)
        # simulates time it takes for the container to stop -> should be max 10 seconds before kube kills it
        yield from replica.simulator.teardown(self.env, replica)

        self.env.cluster.remove_pod_from_node(replica.pod, node)
        replica.state = FunctionState.SUSPENDED
        self.lb_replicas[replica.function.name].remove(replica)
        self.env.metrics.log('allocation', {
            'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
            'mem': 1 - (node.allocatable.memory / node.capacity.memory)
        }, node=node.name)

        self.lb_replica_count[replica.fn_name] -= 1
        self.lb_replica_per_image_count[replica.image] -= 1

    def run_lb_scheduler_worker(self):
        pass


class LocalizedLoadBalancerFaasSystem(LoadBalancerCapableFaasSystem):
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
