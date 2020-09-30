from collections import defaultdict
from typing import Dict

import pandas as pd

from faassim.logging import RuntimeLogger, NullLogger
from sim.core import Environment


class Metrics:
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

    def log_function(self, fn):
        """
        Logs the functions name, related container images and their metadata
        """
        for container in fn.pod.spec.containers:
            record = {'name': fn.name, 'pod': fn.pod.name, 'image': container.image}
            image_state = self.env.cluster.image_states[container.image]
            for arch, size in image_state.size.items():
                record[f'size_{arch}'] = size

            self.log('functions', record)

    def log_flow(self, num_bytes, duration, source, sink, action_type):
        self.log('flow', value={'bytes': num_bytes, 'duration': duration},
                             source=source.name, sink=sink.name, action_type=action_type)

    def log_network(self, num_bytes, data_type, link):
        tags = dict(link.tags)
        tags['data_type'] = data_type

        self.log('network', num_bytes, **tags)

    def log_scaling(self, function_name, replicas):
        self.log('scale', replicas, function_name=function_name)

    def log_invocation(self, function_name, node_name, t_wait, t_exec):
        self.invocations[function_name] += 1
        self.total_invocations += 1
        self.last_invocation[function_name] = self.env.now

        function = self.env.faas.functions[function_name]
        mem = function.get_resource_requirements().get('memory')

        self.log('invocations', {'t_wait': t_wait, 't_exec': t_exec, 'memory': mem},
                             function_name=function_name, node=node_name)

    def log_start_exec(self, request, replica):
        node = replica.node
        function = replica.function

        for resource, value in function.get_resource_requirements().items():
            self.utilization[node.name][resource] += value

        self.log('utilization', {
            'cpu': self.utilization[node.name]['cpu'] / node.capacity.cpu_millis,
            'mem': self.utilization[node.name]['memory'] / node.capacity.memory
        }, node=node.name)

    def log_stop_exec(self, request, replica):
        node = replica.node
        function = replica.function

        for resource, value in function.get_resource_requirements().items():
            self.utilization[node.name][resource] -= value

        self.log('utilization', {
            'cpu': self.utilization[node.name]['cpu'] / node.capacity.cpu_millis,
            'mem': self.utilization[node.name]['memory'] / node.capacity.memory
        }, node=node.name)

    def get(self, name, **tags):
        return self.logger.get(name, **tags)

    @property
    def clock(self):
        return self.clock

    @property
    def records(self):
        return self.logger.records

    def extract_dataframe(self, measurement: str):
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
        df.index = pd.DatetimeIndex(pd.to_datetime(df['time']))
        del df['time']
        return df