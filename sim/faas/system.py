import logging
import time
import uuid
from typing import Dict, List, Generator, Optional

import simpy
from ether.util import parse_size_string
from faas.context import NodeService
from faas.system.core import FunctionContainer, FunctionRequest, FunctionReplicaState, FaasSystem, FunctionResponse

from sim.core import Environment
from sim.faas import RoundRobinLoadBalancer, SimFunctionReplica
from sim.net import LowBandwidthException
from sim.skippy import create_function_pod
from .core import FunctionSimulator, FunctionSimulatorResponse
from ..context.platform.deployment.model import SimFunctionDeployment
from ..context.platform.deployment.service import SimFunctionDeploymentService
from ..context.platform.node.model import SimFunctionNode
from ..context.platform.replica.service import SimFunctionReplicaService
from ..context.platform.trace.service import SimTraceService
from ..factory.flow import FlowFactory

logger = logging.getLogger(__name__)


class DefaultFaasSystem(FaasSystem):
    """
    A default implementation of the FaasSystem interface using faas-sim concepts.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env

        self.request_queue = simpy.Store(env)
        self.scheduler_queue = simpy.Store(env)

        # TODO let users inject SimLoadBalancer
        self.load_balancer = RoundRobinLoadBalancer(env)

    @property
    def replica_service(self) -> SimFunctionReplicaService:
        return self.env.context.replica_service

    @property
    def deployment_service(self) -> SimFunctionDeploymentService:
        return self.env.context.deployment_service

    @property
    def trace_service(self) -> SimTraceService:
        return self.env.context.trace_service

    @property
    def node_service(self) -> NodeService[SimFunctionNode]:
        return self.env.context.node_service

    def get_deployments(self) -> List[SimFunctionDeployment]:
        return self.deployment_service.get_deployments()

    def get_function_index(self) -> Dict[str, FunctionContainer]:
        return self.deployment_service.get_function_containers_by_name()

    def get_replicas(self, fn_name: str, running: bool = True, state=None) -> List[SimFunctionReplica]:
        return self.replica_service.get_function_replicas_of_deployment(fn_name, running, state)

    def deploy(self, fd: SimFunctionDeployment):
        if self.deployment_service.exists(fd.name):
            raise ValueError('function already deployed')

        self.deployment_service.add(fd)
        self.env.metrics.log_function_definition(fd)
        self.env.metrics.log_function_deployment(fd)
        self.env.metrics.log_function_deployment_lifecycle(fd, 'deploy')
        logger.info('deploying function %s with scale_min=%d', fd.name, fd.scaling_config.scale_min)
        yield from self.scale_up(fd.name, fd.scaling_config.scale_min)

    def deploy_replica(self, fd: SimFunctionDeployment, fn: FunctionContainer, services: List[FunctionContainer]):
        """
        Creates and deploys a SimFunctionReplica for the given FunctionContainer.
        In case no node supports the given FunctionContainer, the services list dictates which FunctionContainer to try next.
        In case no FunctionContainer can be hosted, the scheduling process terminates and logs the failed attempt
        """
        replica = self.create_replica(fd, fn)
        self.replica_service.add_function_replica(replica)
        self.env.metrics.log_queue_schedule(replica)
        self.env.metrics.log_function_replica(replica)
        yield self.scheduler_queue.put((replica, services))

    def invoke(self, request: FunctionRequest):
        # TODO: how to return a FunctionResponse?
        logger.debug('invoking function %s', request.name)

        if not self.deployment_service.exists(request.name):
            logger.warning('invoking non-existing function %s', request.name)
            return

        t_received = self.env.now
        if request.replica is None:
            replicas = self.get_replicas(request.name, state=FunctionReplicaState.RUNNING)
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
            elif len(replicas) > 1:
                logger.debug('asking load balancer for replica for request %s:%d', request.name, request.request_id)
                replica = self.next_replica(request)
            else:
                replica = replicas[0]
        else:
            replica = request.replica

        logger.debug('dispatching request %s:%d to %s', request.name, request.request_id, replica.node.name)

        t_start = self.env.now
        response: FunctionResponse = yield from self.simulate_function_invocation(self.env, replica, request)

        t_end = self.env.now
        kwargs = {
            'status': response.code
        }
        if request.client is not None:
            kwargs['client'] = request.client

        self.env.metrics.log_invocation(request.name, replica.image, replica.node.name, t_received, t_start,
                                        t_end, replica.replica_id, request.request_id, **kwargs)
        return response

    def remove(self, fn: SimFunctionDeployment):
        self.env.metrics.log_function_deployment_lifecycle(fn, 'remove')
        self.deployment_service.remove(fn.name)
        replicas: List[SimFunctionReplica] = [f for f in self.replica_service.get_function_replicas() if
                                              f.fn_name == fn.name]
        yield from self.scale_down(fn.name, len(replicas))

    def scale_down(self, fn_name: str, remove: int):
        replica_count = len(self.get_replicas(fn_name, state=FunctionReplicaState.RUNNING))
        if replica_count == 0:
            return
        replica_count -= remove
        if replica_count <= 0:
            remove = remove + replica_count

        scale_min = self.deployment_service.get_by_name(fn_name).scaling_config.scale_min
        if replica_count - remove < scale_min:
            remove = replica_count - scale_min

        if replica_count - remove <= 0 or remove == 0:
            return

        logger.info(f'scale down {fn_name} by {remove}')
        replicas = self.choose_replicas_to_remove(fn_name, remove)
        self.env.metrics.log_scaling(fn_name, -remove)
        for replica in replicas:
            yield from self._remove_replica(replica)

    def choose_replicas_to_remove(self, fn_name: str, n: int):
        # TODO implement more sophisticated, currently just picks last ones deployed
        running_replicas = self.get_replicas(fn_name, state=FunctionReplicaState.RUNNING)
        return running_replicas[len(running_replicas) - n:]

    def scale_up(self, fn_name: str, replicas: int):
        fd = self.deployment_service.get_by_name(fn_name)
        config = fd.scaling_config
        ranking = fd.ranking

        scale = replicas
        fn_replicas = self.get_replicas(fn_name, state=FunctionReplicaState.RUNNING)
        replica_count = len(fn_replicas)

        if replica_count >= config.scale_max:
            logger.debug('Function %s wanted to scale up, but maximum number of replicas reached', fn_name)
            return

        # check whether request would exceed maximum number of containers for the function and reduce to scale to max
        if replica_count + replicas > config.scale_max:
            reduce = replica_count + replicas - config.scale_max
            scale = replicas - reduce

        if scale == 0:
            return
        actually_scaled = 0
        for index, service in enumerate(fd.get_services()):
            # check whether service has capacity, otherwise continue
            leftover_scale = scale
            max_replicas = int(ranking.function_factor[service.image] * config.scale_max)
            fn_container_count = len([f for f in fn_replicas if f.container.image == service.image])
            # check if scaling all new pods would exceed the maximum number of replicas for this function container
            if max_replicas * config.scale_max < leftover_scale + fn_container_count:

                # calculate how many pods of this service can be deployed while satisfying the max function factor
                reduce = max_replicas - fn_container_count
                if reduce < 0:
                    # all replicas used
                    continue
                leftover_scale = leftover_scale - reduce
            if leftover_scale > 0:
                for _ in range(leftover_scale):
                    yield from self.deploy_replica(fd, fd.get_container(service.image), fd.get_containers()[index:])
                    actually_scaled += 1
                    scale -= 1

        self.env.metrics.log_scaling(fd.name, actually_scaled)

        if scale > 0:
            logger.debug("Function %s wanted to scale, but not all requested replicas were deployed: %s", fn_name,
                         str(scale))

    def next_replica(self, request) -> SimFunctionReplica:
        return self.load_balancer.next_replica(request)

    def start(self):
        for process in self.env.background_processes:
            self.env.process(process(self.env))
        self.env.process(self.run_scheduler_worker())

    def poll_available_replica(self, fn: str, interval=0.5, timeout: int = None):
        replicas = self.get_replicas(fn, state=FunctionReplicaState.RUNNING)
        while len(replicas) == 0:
            yield self.env.timeout(interval)
            replicas = self.get_replicas(fn, state=FunctionReplicaState.RUNNING)

    def run_scheduler_worker(self):
        env = self.env

        while True:
            replica: SimFunctionReplica
            replica, services = yield self.scheduler_queue.get()

            logger.debug('scheduling next replica %s', replica.function.name)

            # schedule the required pod
            self.env.metrics.log_start_schedule(replica)
            pod = replica.pod
            then = time.time()
            result = env.scheduler.schedule(pod)
            duration = time.time() - then
            self.env.metrics.log_finish_schedule(replica, result)

            yield env.timeout(duration)  # include scheduling latency in simulation time

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Pod scheduling took %.2f ms, and yielded %s', duration * 1000, result)

            if not result.suggested_host:
                self.replica_service.delete_function_replica(replica.replica_id)
                if len(services) > 0:
                    logger.warning('retry scheduling pod %s', pod.name)
                    yield from self.deploy_replica(replica.function, services[0], services[1:])
                else:
                    logger.error('pod %s cannot be scheduled', pod.name)

                continue

            logger.info('pod %s was scheduled to %s', pod.name, result.suggested_host)

            replica.node = self.node_service.find(result.suggested_host.name)
            node = replica.node.skippy_node

            env.metrics.log('allocation', {
                'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
                'mem': 1 - (node.allocatable.memory / node.capacity.memory)
            }, node=node.name)

            # TODO check if replica is set to RUNNING immediately, actually must be PENDING
            # self.functions_definitions[replica.image] += 1
            # self.replica_count[replica.fn_name] += 1

            self.env.metrics.log_function_deploy(replica)
            # start a new process to simulate starting of pod

            # TODO Replica must still be PENDING and in the simulate_function_start is has to be set to running
            env.process(simulate_function_start(env, replica))

    def create_pod(self, fd: SimFunctionDeployment, fn: FunctionContainer):
        return create_function_pod(fd, fn)

    def create_replica(self, fd: SimFunctionDeployment, fn: FunctionContainer) -> SimFunctionReplica:
        replica_id = fd.name + '-' + str(uuid.uuid4())[12:].replace('-', '')
        replica = SimFunctionReplica(replica_id, fn.labels, fd, fn, None, FunctionReplicaState.CONCEIVED)
        replica.function = fd
        replica.container = fn
        replica.pod = self.create_pod(fd, fn)
        # this simulator dictates the behavior of the replica
        replica.simulator = self.env.simulator_factory.create(self.env, fn)
        return replica

    def discover(self, function: str, **kwargs) -> List[SimFunctionReplica]:
        return self.replica_service.get_function_replicas_of_deployment(function, running=True)

    def _remove_replica(self, replica: SimFunctionReplica):
        env = self.env

        # set the state to SHUTDOWN
        self.replica_service.shutdown_function_replica(replica.replica_id)
        env.metrics.log_teardown(replica)

        node = replica.node.skippy_node

        yield from replica.simulator.teardown(env, replica)

        self.env.cluster.remove_pod_from_node(replica.pod, node)

        # set the state to DELETE
        self.replica_service.delete_function_replica(replica.replica_id)

        env.metrics.log('allocation', {
            'cpu': 1 - (node.allocatable.cpu_millis / node.ether_node.capacity.cpu_millis),
            'mem': 1 - (node.allocatable.memory / node.ether_node.capacity.memory)
        }, node=node.name)

    def suspend(self, function_name: str):
        deployment = self.deployment_service.get_by_name(function_name)
        if deployment is None:
            raise ValueError

        # TODO interrupt startup of function containers that are starting
        replicas: List[SimFunctionReplica] = [f for f in self.replica_service.get_function_replicas() if
                                              f.fn_name == function_name]
        self.scale_down(function_name, len(replicas))

        self.env.metrics.log_function_deployment_lifecycle(deployment, 'suspend')

    def simulate_function_invocation(self, env: Environment, replica: SimFunctionReplica,
                                     request: FunctionRequest) -> Generator[None, None, FunctionResponse]:
        t_start = env.now
        env.metrics.log_start_exec(request, replica)

        # simulate transfer of request from invoker to function
        if request.client is not None and request.size is not None:
            src_name = request.client
            dest_name = replica.node.name
            size_kb = request.size
            success = yield from self.simulate_request_transfer(self.env.flow_factory, request.name, request.request_id,
                                                                src_name,
                                                                dest_name, size_kb)
            if not success:
                logger.info(
                    f'A LowBandwidthException occurred when calling {request.name} with the client'
                    f'{request.client} and the destination node {replica.node.name}.'
                    f'The Call is aborted.')
                return 500

        simulator: FunctionSimulator = replica.simulator
        # if sim response is None, this will be used for t_exec and t_wait
        t_wait = env.now
        t_exec = env.now
        sim_response: Optional[FunctionSimulatorResponse] = yield from simulator.invoke(env, replica, request)
        t_exec_end = env.now
        code = 200
        size = None
        body = ''
        # default fet calculation, only considered when sim response is None
        fet = t_exec_end - t_exec

        if sim_response is not None:
            code = sim_response.code
            size = sim_response.size
            body = sim_response.body
            t_wait = sim_response.t_wait
            t_exec = sim_response.t_exec
            fet = sim_response.fet

            if request.client is not None and sim_response.size is not None:
                # simulate transfer of response from function to invoker
                src_name = replica.node.name
                dest_name = request.client
                size_kb = sim_response.size
                success = yield from self.simulate_request_transfer(self.env.flow_factory, request.name,
                                                                    request.request_id,
                                                                    src_name, dest_name, size_kb)

                if not success:
                    logger.info(
                        f'A LowBandwidthException occurred when returning the response of {request.name} with the client'
                        f'{request.client} and the destination node {replica.node.name}.'
                        f'The Call is aborted.')
                    code = 500
        else:
            logger.debug(f'{request.request_id} go None response.')
        env.metrics.log_stop_exec(request, replica)
        t_end = env.now

        response = FunctionResponse(
            request,
            request_id=request.request_id,
            client=request.client,
            name=request.name,
            body=body,
            code=code,
            t_start=t_start,
            t_end=t_end,
            replica=replica,
            node=replica.node,
            t_wait=t_wait,
            t_exec=t_exec,
            size=size,
            fet=fet
        )
        env.context.trace_service.add_trace(response)
        return response

    def simulate_request_transfer(self, flow_factory: FlowFactory, fn_name: str, request_id: int, src_name: str,
                                  dest_name: str,
                                  size_kb: int) -> Generator[
        None, None, bool]:
        """
        Simulates a transfer from src to dest. The size is in KB.
        The return value of this function indicates the result of the transfer.
        If it is True, the transfer happened.
        If it is False, the transfer failed due to LowBandwidth
        """
        # TODO this is missing the load balancer, see TODO at the bottom of file
        env = self.env
        started = env.now
        route = self.env.topology.route_by_node_name(src_name, dest_name)
        if len(route.hops) == 0:
            return True
        try:
            flow = flow_factory.create_flow(self.env, size_kb * 1024, route)
            yield flow.start()
            for hop in route.hops:
                env.metrics.log_network(size_kb, fn_name, hop, request_id=request_id)
            env.metrics.log_flow(size_kb, env.now - started, route.source, route.destination, fn_name,
                                 request_id=request_id)
            return True
        except LowBandwidthException:
            logger.debug(f'LowBandwidthException thrown while transferring {size_kb}KB from {src_name} to {dest_name}')
            return False


def simulate_function_start(env: Environment, replica: SimFunctionReplica):
    sim: FunctionSimulator = replica.simulator
    # TODO check if mutating the state here causes problems
    # TODO this function can be interrupted through suspend, i.e., a function starting up might be shutdown the next moment
    logger.debug('deploying function %s to %s', replica.function.name, replica.node.name)
    env.metrics.log_deploy(replica)
    yield from sim.deploy(env, replica)
    replica.state = FunctionReplicaState.PENDING
    env.metrics.log_startup(replica)
    logger.debug('starting function %s on %s', replica.function.name, replica.node.name)
    yield from sim.startup(env, replica)

    logger.debug('running function setup %s on %s', replica.function.name, replica.node.name)
    env.metrics.log_setup(replica)
    yield from sim.setup(env, replica)  # FIXME: this is really domain-specific startup
    env.metrics.log_finish_deploy(replica)
    replica.state = FunctionReplicaState.RUNNING


def simulate_data_download(env: Environment, replica: SimFunctionReplica):
    node = replica.node.ether_node
    func = replica
    started = env.now

    if 'data.skippy.io/receives-from-storage' not in func.pod.spec.labels:
        return

    # FIXME: storage
    size = parse_size_string(func.pod.spec.labels['data.skippy.io/receives-from-storage'])
    path = func.pod.spec.labels['data.skippy.io/receives-from-storage/path']

    storage_node_name = env.cluster.get_storage_nodes(path)[0]
    logger.debug('%.2f replica %s fetching data %s from %s', env.now, node, path, storage_node_name)

    if storage_node_name == node.name:
        # FIXME this is essentially a disk read and not a network connection
        yield env.timeout(size / 1.25e+8)  # 1.25e+8 = 1 GBit/s
        return

    storage_node = env.cluster.get_node(storage_node_name)
    route = env.topology.route_by_node_name(storage_node.name, node.name)
    flow = env.flow_factory.create_flow(env, size, route)
    yield flow.start()
    for hop in route.hops:
        env.metrics.log_network(size, 'data_download', hop)
    env.metrics.log_flow(size, env.now - started, route.source, route.destination, 'data_download')


def simulate_data_upload(env: Environment, replica: SimFunctionReplica):
    node = replica.node.ether_node
    func = replica
    started = env.now

    if 'data.skippy.io/sends-to-storage' not in func.pod.spec.labels:
        return

    # FIXME: storage
    size = parse_size_string(func.pod.spec.labels['data.skippy.io/sends-to-storage'])
    path = func.pod.spec.labels['data.skippy.io/sends-to-storage/path']

    storage_node_name = env.cluster.get_storage_nodes(path)[0]
    logger.debug('%.2f replica %s uploading data %s to %s', env.now, node, path, storage_node_name)

    if storage_node_name == node.name:
        # FIXME this is essentially a disk read and not a network connection
        yield env.timeout(size / 1.25e+8)  # 1.25e+8 = 1 GBit/s
        return

    storage_node = env.cluster.get_node(storage_node_name)
    route = env.topology.route_by_node_name(node.name, storage_node.name)
    flow = env.flow_factory.create_flow(env, size, route)
    yield flow.start()
    for hop in route.hops:
        env.metrics.log_network(size, 'data_upload', hop)
    env.metrics.log_flow(size, env.now - started, route.source, route.destination, 'data_upload')

# TODO the following implementation of jacob's thesis branch should be adopted as it explictly transfers data from the client
# to the load balancer and from the load balancer to the function - which is how it happens in real life
# def simulate_function_invocation(env, replica: FunctionReplica, request: FunctionRequest):
#     """
#     Adapted version of "simulate function invocation" that also includes network simulation
#     @param replica: The function replica
#     @param request: The request to be processed
#     """
#     tx_time_cl_lb = 0
#     tx_time_lb_fx = 0
#     t_start = self.env.now
#
#     if request.load_balancer is not None and hasattr(request, 'client_node') and isinstance(request.load_balancer,
#                                                                                             LocalizedLoadBalancer):
#         # right now I used 250kb request payload, which should be a small JPG with the added HTTP overhead
#         cl_lb_start = self.env.now
#         yield from self.simulate_request_transfer(request.load_balancer.ether_node.name, request.client_node.name,
#                                                   250)
#         tx_time_cl_lb = self.env.now - cl_lb_start
#         lb_fx_start = self.env.now
#         yield from self.simulate_request_transfer(request.load_balancer.ether_node.name,
#                                                   replica.node.ether_node.name,
#                                                   250)
#         tx_time_lb_fx = self.env.now - lb_fx_start
#
#     # actual function simulation portion
#     yield from simulate_function_invocation(self.env, replica, request)
#     t_end = self.env.now
#
#     # report the total response time to the load-balancer for weight updates etc.
#     if request.load_balancer is not None:
#         if isinstance(request.load_balancer, LeastResponseTimeLoadBalancer):
#             request.load_balancer.report_response_time(request, replica, t_end - t_start)
#
#     if tx_time_cl_lb != 0 and tx_time_lb_fx != 0:
#         request.tx_time_cl_lb = tx_time_cl_lb
#         request.tx_time_lb_fx = tx_time_lb_fx
