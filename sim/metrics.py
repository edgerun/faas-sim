import glob
import os.path
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from ether.core import Capacity
from faas.system.core import FunctionRequest
from skippy.core.clustercontext import ClusterContext
from skippy.core.model import SchedulingResult

from faas.system import Metrics, RuntimeLogger, NullLogger, MetricsLogger, WallClock, Record
from sim.context.platform.deployment.model import SimFunctionDeployment
from sim.core import Environment
from sim.faas import SimFunctionReplica


class FlushingRuntimeLogger(MetricsLogger):
    """This implementation continuously extracts all data into dataframes and flushes the recorded
    logs. While it is not possible extract the dataframes after the simulation finished, it increases the scalability of the simulation by keeping the memory usage constant for the metrics collection."""

    def __init__(self, results_folder: str, measurement_keys: List[str] = None, buffer_size: int = 100,
                 clock=None) -> None:
        self.results_folder = results_folder
        self._init_results_folder()
        self.records = list()
        self.buffer_size = buffer_size
        if measurement_keys is None:
            self.measurement_keys = ['allocation',
                                     'invocations',
                                     'scale',
                                     'schedule',
                                     'replica_deployment',
                                     'function_deployments',
                                     'function_deployment',
                                     'function_deployment_lifecycle',
                                     'functions',
                                     'flow',
                                     'network',
                                     'node_utilization',
                                     'function_utilization',
                                     'fets']
        else:
            self.measurement_keys = measurement_keys
        self.clock = clock or WallClock()

    def _init_results_folder(self):
        Path(self.results_folder).mkdir(parents=True, exist_ok=False)

    def get(self, name, **tags):
        return lambda x: self.log(name, x, None, **tags)

    def log(self, metric, value, time=None, **tags):
        """
        Call l.log('cpu_load', .65, host='server0', region='us-west') or

        :param metric: the name of the measurement
        :param value: the measurement value
        :param time: the (optional) time, otherwise now will be used
        :param tags: additional tags describing the measurement
        :return:
        """
        if time is None:
            time = self._now()

        if type(value) == dict:
            fields = value
        else:
            fields = {
                'value': value
            }

        self._store_record(Record(metric, time, fields, tags))

    def _store_record(self, record: Record):
        self.records.append(record)
        if len(self.records) > self.buffer_size:
            self._flush()

    def _flush(self):

        new_measurement_dfs = {}
        old_measurement_dfs = self._read_existing_dfs()

        for measurement in self.measurement_keys:
            df = extract_dataframe(self.records, measurement)
            if old_measurement_dfs is not None:
                old_df = old_measurement_dfs[measurement]
                new_measurement_dfs[measurement] = pd.concat([old_df, df])
            else:
                new_measurement_dfs[measurement] = df

        self._save_measurements(new_measurement_dfs)
        del self.records
        self.records = list()

    def _save_measurements(self, measurement_dfs: Dict[str, pd.DataFrame]):
        for measurement, df in measurement_dfs.items():
            file_path = f'{self.results_folder}/{measurement}.csv'
            df.to_csv(file_path)

    def _read_existing_dfs(self):
        filenames = glob.glob(f'{self.results_folder}/*.csv')
        if len(filenames) == 0:
            return None
        dfs = {}
        for filename in filenames:
            df = pd.read_csv(filename)
            measurement_key = os.path.basename(filename).replace('.csv', '')
            dfs[measurement_key] = df

        return dfs

    def _now(self):
        return self.clock.now()


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

    def log_function_resource_utilization(self, replica: SimFunctionReplica, utilization: pd.DataFrame):
        node = replica.node
        resources = utilization['resource'].unique()
        for resource in resources:
            mean = utilization[utilization['resource'] == resource]['value'].mean()
            if mean is not None and len(utilization) > 0:
                self.log('function_utilization', mean, resource=resource, node=node.name, replica_id=replica.replica_id)

    def log_resource_utilization(self, node_name: str, capacity: Capacity, utilization: pd.DataFrame):
        if len(utilization) != 0:
            resources = utilization['resource'].unique()
            for resource in resources:
                mean = utilization[utilization['resource'] == resource]['value'].mean()
                self.log('node_utilization', mean, resource=resource, node=node_name)

    def __calculate_util(self, capacity, utilization):
        # update = {
        #     'cpu_util': utilization.get_resource('cpu') / capacity.cpu_millis if utilization.get_resource(
        #         'cpu') is not None else 0,
        #     'mem_util': utilization.get_resource('memory') / capacity.memory if utilization.get_resource(
        #         'memory') is not None else 0
        # }
        # resources = utilization.list_resources()
        # resources.update(update)
        # cpu = utilization[utilization['resource'] == 'cpu']
        # return cpu['value'].mean()
        # TODO update me to use the dataframe
        pass

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

    def extract_dataframe(self, measurement: str):
        return extract_dataframe(self.records, measurement)


def extract_dataframe(records, measurement: str):
    if measurement == 'traces':
        return _get_traces(records)

    data = list()

    for record in records:
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


def _get_traces(records):
    requests_by_id = defaultdict(list)
    for record in records:
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
        r['t_exec'] = t_exec
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
