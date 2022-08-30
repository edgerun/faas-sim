from collections import defaultdict
from typing import Dict

import pandas as pd
from ether.core import Capacity
from faas.system import Metrics, RuntimeLogger, NullLogger
from faas.system.core import FunctionRequest
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import SchedulingResult

from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.faas import SimFunctionReplica
from sim.resource import ResourceUtilization


class SimMetrics(Metrics):
    """
    Instrumentation and trace logger.
    """
    invocations: Dict[str, int]
    total_invocations: int
    last_invocation: Dict[str, float]
    utilization: Dict[str, Dict[str, float]]

    def __init__(self, env: Environment, log: RuntimeLogger = None) -> None:
        super().__init__()
        self.env: Environment = env
        self.logger: RuntimeLogger = log or NullLogger()
        self.total_invocations = 0
        self.invocations = defaultdict(int)
        self.last_invocation = defaultdict(int)
        self.utilization = defaultdict(lambda: defaultdict(float))

    def log(self, metric, value, **tags):
        return self.logger.log(metric, value, **tags)

    def log_function_deployment(self, fn: SimFunctionDeployment):
        """
        Logs the functions name, related container images and their metadata
        """
        record = {'name': fn.name, 'scale_min': fn.scaling_configuration.scale_min,
                  'scale_max': fn.scaling_configuration.scale_max,
                  'scale_factor': fn.scaling_configuration.scale_factor,
                  'scale_zero': fn.scaling_configuration.scale_zero}
        self.log('function_deployments', record)

    def log_function_definition(self, fn: SimFunctionDeployment):
        record = {'name': fn.fn.name, 'labels': fn.fn.labels}
        self.log('functions', record)

    def log_function_image_definitions(self, fn: SimFunctionDeployment):
        for function_image in fn.fn.fn_images:
            record = {'function_name': fn.name, 'image': function_image.image}
            self.log('function_images', record)

    def log_function_container_definitions(self, fn: SimFunctionDeployment):
        """
        Logs the functions name, related container images and their metadata
        """

        cluster: ClusterContext = self.env.cluster
        for fn_container in fn.fn_containers:
            record = {'name': fn.name, 'image': fn_container.image, 'sizes': {},
                      'resource_requirements': fn_container.get_resource_requirements(), 'labels': fn_container.labels}

            image_state = cluster.retrieve_image_state(fn_container.fn_image.image)
            for arch, size in image_state.size.items():
                record['sizes'][f'size_{arch}'] = size

            self.log('function_containers', record)

    def log_function_replica(self, replica: SimFunctionReplica, **kwargs):
        cpu_req = replica.container.get_resource_requirements()['cpu']
        memory_req = replica.container.get_resource_requirements()['memory']
        for container in replica.pod.spec.containers:
            record = {'name': replica.function.name, 'pod': replica.pod.name, 'image': container.image,
                      'cpu_request': cpu_req, 'state': replica.state.name,
                      'memory_req': memory_req, 'labels': replica.labels}

            self.log('function_replicas', record, replica_id=replica.replica_id, **kwargs)

    def log_flow(self, num_bytes, duration, source, sink, action_type, **kwargs):
        self.log('flow', value={'bytes': num_bytes, 'duration': duration},
                 source=source.name, sink=sink.name, action_type=action_type, **kwargs)

    def log_network(self, num_bytes, data_type, link, **kwargs):
        tags = dict(link.tags)
        tags['data_type'] = data_type

        self.log('network', num_bytes, **tags, **kwargs)

    def log_scaling(self, function_name, replicas, **kwargs):
        self.log('scale', replicas, function_name=function_name, **kwargs)

    def log_invocation(self, function_name, function_image, node_name, ts_received, ts_start, ts_end, replica_id,
                       request_id,
                       **kwargs):
        function = self.env.faas.get_function_index()[function_image]
        mem = function.get_resource_requirements().get('memory')

        self.log('invocations',
                 {'t_wait': ts_received - ts_start, 'ts_received': ts_received, 't_exec': ts_end - ts_start,
                  'ts_end': ts_end,
                  'ts_start': ts_start, 'memory': mem, **kwargs},
                 function_name=function_name,
                 function_image=function_image, node=node_name, replica_id=replica_id, request_id=request_id)

    def log_fet(self, replica: SimFunctionReplica, request: FunctionRequest, ts_fet_start, ts_fet_end,
                **kwargs):
        function_name = replica.fn_name
        function_image = replica.image
        node_name = replica.node.name
        replica_id = replica.replica_id
        request_id = request.request_id
        self.log('fets', {'ts_fet_start': ts_fet_start, 'ts_fet_end': ts_fet_end, **kwargs},
                 function_name=function_name,
                 function_image=function_image, node=node_name, replica_id=replica_id, request_id=request_id)

    def log_function_resource_utilization(self, replica: SimFunctionReplica, utilization: ResourceUtilization):
        node = replica.node
        copy = utilization.copy()
        resources = self.__calculate_util(node.ether_node.capacity, copy)
        self.log('function_utilization', resources, node=node.name, replica_id=replica.replica_id)

    def log_resource_utilization(self, node_name: str, capacity: Capacity, utilization: ResourceUtilization):
        resources = self.__calculate_util(capacity, utilization)
        self.log('node_utilization', resources, node=node_name)

    def __calculate_util(self, capacity, utilization):
        update = {
            'cpu_util': utilization.get_resource('cpu') / capacity.cpu_millis if utilization.get_resource(
                'cpu') is not None else 0,
            'mem_util': utilization.get_resource('memory') / capacity.memory if utilization.get_resource(
                'memory') is not None else 0
        }
        resources = utilization.list_resources()
        resources.update(update)
        return resources

    def log_start_exec(self, request: FunctionRequest, replica: SimFunctionReplica, **kwargs):
        self.invocations[replica.function.name] += 1
        self.total_invocations += 1
        self.last_invocation[replica.function.name] = self.env.now

    def log_stop_exec(self, request: FunctionRequest, replica: SimFunctionReplica, **kwargs):
        pass

    def log_deploy(self, replica: SimFunctionReplica):
        self.log('replica_deployment', 'deploy', function_name=replica.function.name, node_name=replica.node.name,
                 image=replica.image, replica_id=replica.replica_id)

    def log_startup(self, replica: SimFunctionReplica):
        self.log('replica_deployment', 'startup', function_name=replica.function.name, node_name=replica.node.name,
                 image=replica.image, replica_id=replica.replica_id)

    def log_setup(self, replica: SimFunctionReplica):
        self.log('replica_deployment', 'setup', function_name=replica.function.name, node_name=replica.node.name,
                 image=replica.image, replica_id=replica.replica_id)

    def log_finish_deploy(self, replica: SimFunctionReplica):
        self.log('replica_deployment', 'finish', function_name=replica.function.name, node_name=replica.node.name,
                 image=replica.image, replica_id=replica.replica_id)

    def log_teardown(self, replica: SimFunctionReplica):
        name = replica.fn_name
        node_name = replica.node.name
        self.log('replica_deployment', 'teardown', function_name=name, node_name=node_name, image=replica.image,
                 replica_id=replica.replica_id)

    def log_delete(self, replica: SimFunctionReplica):
        name = replica.fn_name
        node_name = replica.node.name
        self.log('replica_deployment', 'delete', function_name=name, node_name=node_name, image=replica.image,
                 replica_id=replica.replica_id)

    def log_function_deployment_lifecycle(self, fn: SimFunctionDeployment, event: str):
        self.log('function_deployment_lifecycle', event, name=fn.name, function_id=id(fn))

    def log_queue_schedule(self, replica: SimFunctionReplica):
        name = replica.fn_name
        image = replica.image
        self.log('schedule', 'queue', function_name=name, image=image,
                 replica_id=replica.replica_id)

    def log_start_schedule(self, replica: SimFunctionReplica):
        name = replica.fn_name
        image = replica.image
        self.log('schedule', 'start', function_name=name, image=image,
                 replica_id=replica.replica_id)

    def log_finish_schedule(self, replica: SimFunctionReplica, result: SchedulingResult):
        if not result.suggested_host:
            node_name = 'None'
        else:
            node_name = result.suggested_host.name

        self.log('schedule', 'finish', function_name=replica.function.name, image=replica.container.image,
                 node_name=node_name,
                 successful=node_name != 'None', replica_id=replica.replica_id)

    def get(self, name, **tags):
        return self.logger.get(name, **tags)

    @property
    def clock(self):
        return self.clock

    @property
    def records(self):
        return self.logger.records

    def _get_traces(self):
        requests_by_id = defaultdict(list)
        for record in self.records:
            if record.measurement == 'invocations':
                for k, v in record.fields.items():
                    if k == 'request_id':
                        requests_by_id[v].append(record)
                for k, v in record.tags.items():
                    if k == 'request_id':
                        requests_by_id[v].append(record)
        data = list()

        for request_id, records in requests_by_id.items():
            max_rtt = 0
            max_response = None
            last_start = 0
            last_response = None
            for record in records:
                if record.fields['t_exec'] > max_rtt:
                    # this is the invocation of the client to load balancer
                    max_rtt = record.fields['t_exec']
                    max_response = record
                if record.fields['ts_start'] > last_start:
                    # this is the last invocation from load balancer to actual replica
                    last_start = record.fields['ts_start']
                    last_response = record
            t_wait = max_response.fields['t_wait']
            ts_received = max_response.fields['ts_received']
            t_exec = last_response.fields['t_exec']
            ts_end = last_response.fields['ts_end']
            ts_start = max_response.fields['ts_start']
            memory = last_response.fields['memory']
            status = max_response.fields['status']
            if max_response.fields.get('client'):
                client = max_response.fields['client']
            else:
                client = 'N/A'

            function_name = last_response.tags['function_name']
            function_image = last_response.tags['function_image']
            node = last_response.tags['node']
            replica_id = last_response.tags['replica_id']
            request_id = last_response.tags['request_id']

            r = dict()
            r['time'] = max_response.time
            r['t_wait'] = t_wait
            r['ts_received'] = ts_received
            r['ts_exec'] = t_exec
            r['ts_end'] = ts_end
            r['ts_start'] = ts_start
            r['memory'] = memory
            r['status'] = status
            r['client'] = client
            r['function_name'] = function_name
            r['function_image'] = function_image
            r['node'] = node
            r['replica_id'] = replica_id
            r['request_id'] = request_id
            data.append(r)

        df = pd.DataFrame(data)

        if len(data) == 0:
            return df

        df.index = pd.DatetimeIndex(pd.to_datetime(df['time']))
        del df['time']
        return df

    def extract_dataframe(self, measurement: str):
        if measurement == 'traces':
            return self._get_traces()

        data = list()

        for record in self.records:
            if record.measurement != measurement:
                continue

            r = dict()
            r['time'] = record.time
            for k, v in record.fields.items():
                r[k] = v
            for k, v in record.tags.items():
                r[k] = v

            data.append(r)
        df = pd.DataFrame(data)

        if len(data) == 0:
            return df

        df.index = pd.DatetimeIndex(pd.to_datetime(df['time']))
        del df['time']
        return df
