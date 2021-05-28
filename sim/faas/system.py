import logging
import time
from collections import defaultdict, Counter
from typing import Dict, List

import simpy
from ether.util import parse_size_string

from sim.core import Environment
from sim.faas import RoundRobinLoadBalancer, FunctionDeployment, FunctionReplica, FunctionContainer, FunctionRequest, \
    FunctionState
from sim.net import SafeFlow
from sim.skippy import create_function_pod
from .core import FaasSystem, FunctionSimulator
from .scaling import FaasRequestScaler, AverageFaasRequestScaler, AverageQueueFaasRequestScaler

logger = logging.getLogger(__name__)


class DefaultFaasSystem(FaasSystem):
    """
    A default implementation of the FaasSystem interface using faas-sim concepts.
    """

    # TODO probably best to inject scaler via env as backgroundprocess - these scalers need to handle all deployed functions
    # currently a scaler per function deployment is started
    def __init__(self, env: Environment, scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False) -> None:
        self.env = env
        self.function_containers = dict()
        # collects all FunctionReplicas under the name of the corresponding FunctionDeployment
        self.replicas = defaultdict(list)

        self.request_queue = simpy.Store(env)
        self.scheduler_queue = simpy.Store(env)

        # TODO let users inject LoadBalancer
        self.load_balancer = RoundRobinLoadBalancer(env, self.replicas)

        self.functions_deployments: Dict[str, FunctionDeployment] = dict()
        self.replica_count: Dict[str, int] = dict()
        self.functions_definitions = Counter()

        self.scale_by_requests = scale_by_requests
        self.scale_by_average_requests_per_replica = scale_by_average_requests
        self.scale_by_queue_requests_per_replica = scale_by_queue_requests_per_replica
        self.faas_scalers: Dict[str, FaasRequestScaler] = dict()
        self.avg_faas_scalers: Dict[str, AverageFaasRequestScaler] = dict()
        self.queue_faas_scalers: Dict[str, AverageQueueFaasRequestScaler] = dict()

    def get_deployments(self) -> List[FunctionDeployment]:
        return list(self.functions_deployments.values())

    def get_function_index(self) -> Dict[str, FunctionContainer]:
        return self.function_containers

    def get_replicas(self, fn_name: str, state=None) -> List[FunctionReplica]:
        if state is None:
            return self.replicas[fn_name]

        return [replica for replica in self.replicas[fn_name] if replica.state == state]

    def deploy(self, fd: FunctionDeployment):
        if fd.name in self.functions_deployments:
            raise ValueError('function already deployed')

        self.functions_deployments[fd.name] = fd
        # TODO remove specific scaling approaches, it's more extendable to let users start scaling technique that iterates over FDs
        self.faas_scalers[fd.name] = FaasRequestScaler(fd, self.env)
        self.avg_faas_scalers[fd.name] = AverageFaasRequestScaler(fd, self.env)
        self.queue_faas_scalers[fd.name] = AverageQueueFaasRequestScaler(fd, self.env)

        if self.scale_by_requests:
            self.env.process(self.faas_scalers[fd.name].run())
        if self.scale_by_average_requests_per_replica:
            self.env.process(self.avg_faas_scalers[fd.name].run())
        if self.scale_by_queue_requests_per_replica:
            self.env.process(self.queue_faas_scalers[fd.name].run())

        for f in fd.fn_containers:
            self.function_containers[f.image] = f

        # TODO log metadata
        self.env.metrics.log_function_deployment(fd)
        self.env.metrics.log_function_deployment_lifecycle(fd, 'deploy')
        logger.info('deploying function %s with scale_min=%d', fd.name, fd.scaling_config.scale_min)
        yield from self.scale_up(fd.name, fd.scaling_config.scale_min)

    def deploy_replica(self, fd: FunctionDeployment, fn: FunctionContainer, services: List[FunctionContainer]):
        """
        Creates and deploys a FunctionReplica for the given FunctionContainer.
        In case no node supports the given FunctionContainer, the services list dictates which FunctionContainer to try next.
        In case no FunctionContainer can be hosted, the scheduling process terminates and logs the failed attempt
        """
        replica = self.create_replica(fd, fn)
        self.replicas[fd.name].append(replica)
        self.env.metrics.log_queue_schedule(replica)
        self.env.metrics.log_function_replica(replica)
        yield self.scheduler_queue.put((replica, services))

    def invoke(self, request: FunctionRequest):
        # TODO: how to return a FunctionResponse?
        logger.debug('invoking function %s', request.name)

        if request.name not in self.functions_deployments.keys():
            logger.warning('invoking non-existing function %s', request.name)
            return

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
        elif len(replicas) > 1:
            logger.debug('asking load balancer for replica for request %s:%d', request.name, request.request_id)
            replica = self.next_replica(request)
        else:
            replica = replicas[0]

        logger.debug('dispatching request %s:%d to %s', request.name, request.request_id, replica.node.name)

        t_start = self.env.now
        yield from simulate_function_invocation(self.env, replica, request)

        t_end = self.env.now

        t_wait = t_start - t_received
        t_exec = t_end - t_start
        self.env.metrics.log_invocation(request.name, replica.image, replica.node.name, t_wait, t_start,
                                        t_exec, id(replica))

    def remove(self, fn: FunctionDeployment):
        self.env.metrics.log_function_deployment_lifecycle(fn, 'remove')

        replica_count = self.replica_count[fn.name]
        yield from self.scale_down(fn.name, replica_count)
        # TODO can be removed after using a central scaler for all FDs
        self.faas_scalers[fn.name].stop()
        self.avg_faas_scalers[fn.name].stop()
        self.queue_faas_scalers[fn.name].stop()

        del self.functions_deployments[fn.name]
        del self.faas_scalers[fn.name]
        del self.avg_faas_scalers[fn.name]
        del self.queue_faas_scalers[fn.name]
        del self.replica_count[fn.name]
        for container in fn.fn_containers:
            del self.functions_definitions[container.image]

    def scale_down(self, fn_name: str, remove: int):
        replica_count = len(self.get_replicas(fn_name, FunctionState.RUNNING))
        if replica_count == 0:
            return
        replica_count -= remove
        if replica_count <= 0:
            remove = remove + replica_count

        scale_min = self.functions_deployments[fn_name].scaling_config.scale_min
        if self.replica_count.get(fn_name, 0) - remove < scale_min:
            remove = self.replica_count.get(fn_name, 0) - scale_min

        if replica_count - remove <= 0 or remove == 0:
            return

        logger.info(f'scale down {fn_name} by {remove}')
        replicas = self.choose_replicas_to_remove(fn_name, remove)
        self.env.metrics.log_scaling(fn_name, -remove)
        for replica in replicas:
            yield from self._remove_replica(replica)
            replicas.remove(replica)

    def choose_replicas_to_remove(self, fn_name: str, n: int):
        # TODO implement more sophisticated, currently just picks last ones deployed
        running_replicas = self.get_replicas(fn_name, FunctionState.RUNNING)
        return running_replicas[len(running_replicas) - n:]

    def scale_up(self, fn_name: str, replicas: int):
        fd = self.functions_deployments[fn_name]
        config = fd.scaling_config
        ranking = fd.ranking

        scale = replicas
        if self.replica_count.get(fn_name, None) is None:
            self.replica_count[fn_name] = 0

        if self.replica_count[fn_name] >= config.scale_max:
            logger.debug('Function %s wanted to scale up, but maximum number of replicas reached', fn_name)
            return

        # check whether request would exceed maximum number of containers for the function and reduce to scale to max
        if self.replica_count[fn_name] + replicas > config.scale_max:
            reduce = self.replica_count[fn_name] + replicas - config.scale_max
            scale = replicas - reduce

        if scale == 0:
            return
        actually_scaled = 0
        for index, service in enumerate(fd.get_services()):
            # check whether service has capacity, otherwise continue
            leftover_scale = scale
            max_replicas = int(ranking.function_factor[service.image] * config.scale_max)

            # check if scaling all new pods would exceed the maximum number of replicas for this function container
            if max_replicas * config.scale_max < leftover_scale + self.functions_definitions[
                service.image]:

                # calculate how many pods of this service can be deployed while satisfying the max function factor
                reduce = max_replicas - self.functions_definitions[service.image]
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

    def next_replica(self, request) -> FunctionReplica:
        return self.load_balancer.next_replica(request)

    def start(self):
        for process in self.env.background_processes:
            self.env.process(process(self.env))
        self.env.process(self.run_scheduler_worker())

    def poll_available_replica(self, fn: str, interval=0.5):
        while not self.get_replicas(fn, FunctionState.RUNNING):
            yield self.env.timeout(interval)

    def run_scheduler_worker(self):
        env = self.env

        while True:
            replica: FunctionReplica
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
                self.replicas[replica.fn_name].remove(replica)
                if len(services) > 0:
                    logger.warning('retry scheduling pod %s', pod.name)
                    yield from self.deploy_replica(replica.function, services[0], services[1:])
                else:
                    logger.error('pod %s cannot be scheduled', pod.name)

                continue

            logger.info('pod %s was scheduled to %s', pod.name, result.suggested_host)

            replica.node = self.env.get_node_state(result.suggested_host.name)
            node = replica.node.skippy_node

            env.metrics.log('allocation', {
                'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
                'mem': 1 - (node.allocatable.memory / node.capacity.memory)
            }, node=node.name)

            self.functions_definitions[replica.image] += 1
            self.replica_count[replica.fn_name] += 1

            self.env.metrics.log_function_deploy(replica)
            # start a new process to simulate starting of pod
            env.process(simulate_function_start(env, replica))

    def create_pod(self, fd: FunctionDeployment, fn: FunctionContainer):
        return create_function_pod(fd, fn)

    def create_replica(self, fd: FunctionDeployment, fn: FunctionContainer) -> FunctionReplica:
        replica = FunctionReplica()
        replica.function = fd
        replica.container = fn
        replica.pod = self.create_pod(fd, fn)
        replica.simulator = self.env.simulator_factory.create(self.env, fn)
        return replica

    def discover(self, function: str) -> List[FunctionReplica]:
        return [replica for replica in self.replicas[function] if replica.state == FunctionState.RUNNING]

    def _remove_replica(self, replica: FunctionReplica):
        env = self.env
        node = replica.node.skippy_node

        env.metrics.log_teardown(replica)
        yield from replica.simulator.teardown(env, replica)

        self.env.cluster.remove_pod_from_node(replica.pod, node)
        replica.state = FunctionState.SUSPENDED
        self.replicas[replica.function.name].remove(replica)

        env.metrics.log('allocation', {
            'cpu': 1 - (node.allocatable.cpu_millis / node.capacity.cpu_millis),
            'mem': 1 - (node.allocatable.memory / node.capacity.memory)
        }, node=node.name)
        self.replica_count[replica.fn_name] -= 1
        self.functions_definitions[replica.image] -= 1

    def suspend(self, function_name: str):
        if function_name not in self.functions_deployments.keys():
            raise ValueError

        # TODO interrupt startup of function containers that are starting
        replicas: List[FunctionReplica] = self.discover(function_name)
        self.scale_down(function_name, len(replicas))

        self.env.metrics.log_function_deployment_lifecycle(self.functions_deployments[function_name], 'suspend')


def simulate_function_start(env: Environment, replica: FunctionReplica):
    sim: FunctionSimulator = replica.simulator

    logger.debug('deploying function %s to %s', replica.function.name, replica.node.name)
    env.metrics.log_deploy(replica)
    yield from sim.deploy(env, replica)
    replica.state = FunctionState.STARTING
    env.metrics.log_startup(replica)
    logger.debug('starting function %s on %s', replica.function.name, replica.node.name)
    yield from sim.startup(env, replica)

    logger.debug('running function setup %s on %s', replica.function.name, replica.node.name)
    env.metrics.log_setup(replica)
    yield from sim.setup(env, replica)  # FIXME: this is really domain-specific startup
    env.metrics.log_finish_deploy(replica)
    replica.state = FunctionState.RUNNING


def simulate_data_download(env: Environment, replica: FunctionReplica):
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
    flow = SafeFlow(env, size, route)
    yield flow.start()
    for hop in route.hops:
        env.metrics.log_network(size, 'data_download', hop)
    env.metrics.log_flow(size, env.now - started, route.source, route.destination, 'data_download')


def simulate_data_upload(env: Environment, replica: FunctionReplica):
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
    flow = SafeFlow(env, size, route)
    yield flow.start()
    for hop in route.hops:
        env.metrics.log_network(size, 'data_upload', hop)
    env.metrics.log_flow(size, env.now - started, route.source, route.destination, 'data_upload')


def simulate_function_invocation(env: Environment, replica: FunctionReplica, request: FunctionRequest):
    env.metrics.log_start_exec(request, replica)
    yield from replica.simulator.invoke(env, replica, request)
    env.metrics.log_stop_exec(request, replica)
