from collections import defaultdict
from typing import Optional, List, Dict

import pandas as pd
from faas.context import TelemetryService
from faas.util.rwlock import ReadWriteLock

from sim.context.platform.replica.model import SimFunctionReplica
from sim.core import Environment
from sim.resource import NodeResourceWindow, ReplicaResourceWindow


class SimTelemetryService(TelemetryService):
    """
    This TelemetryService implementation relies on the ResourceMonitor to continously put
    resource data into this service. The benefit of this implementation is to simulate real-world behavior.
    For example Prometheus must repeatedly pull the data from all notes. Therefore, components accessing resource data
    from Prometheus rely on this pull-process which might not always include the newest data.
    A different implementation of the TelemetryService could simply use the resource state of the
    """

    def __init__(self, window: int, env: Environment):
        self.window = window
        self.env = env
        self._replica_windows: Dict[str, List[ReplicaResourceWindow]] = defaultdict(list)
        self._replicas: Dict[str, SimFunctionReplica] = {}
        self._node_windows: Dict[str, List[NodeResourceWindow]] = defaultdict(list)
        self.rw_lock = ReadWriteLock()
        self.last_update = env.now

    def _cleanup(self):
        now = self.env.now
        start = now - self.window
        replicas_updated = defaultdict(list)
        for replica_id, windows in self._replica_windows.items():
            replicas_updated[replica_id] = [v for v in windows if v.time > start]
        self._replica_windows = replicas_updated

        node_updated = defaultdict(list)
        for node, windows in self._node_windows.items():
            node_updated[node] = [v for v in windows if v.time > start]
        self._node_windows = node_updated

    def put_replica_resource_utilization(self, replica_window: ReplicaResourceWindow):
        with self.rw_lock.lock.gen_wlock():
            replica_id = replica_window.replica.replica_id
            self._replicas[replica_id] = replica_window.replica

            self._replica_windows[replica_id].append(replica_window)

            if self.env.now - self.last_update > self.window:
                self._cleanup()
                self.last_update = self.env.now

    def put_node_resource_utilization(self, node_window: NodeResourceWindow):
        with self.rw_lock.lock.gen_wlock():
            node_name = node_window.node
            self._node_windows[node_name].append(node_window)

            if self.env.now - self.last_update > self.window:
                self._cleanup()
                self.last_update = self.env.now

    def get_replica_cpu(self, fn_replica_id: str, start: int = None, end: int = None) -> \
            Optional[pd.DataFrame]:
        return self.get_replica_resource(fn_replica_id, 'cpu', start, end)

    def get_replica_resource(self, fn_replica_id: str, resource: str, start: int = None, end: int = None) -> Optional[
        pd.DataFrame]:
        with self.rw_lock.lock.gen_rlock():
            resource_windows = self._replica_windows.get(fn_replica_id)
            if resource_windows is None:
                return None

            data = defaultdict(list)
            for resource_window in resource_windows:
                if start is None or resource_window.time > start:
                    if end is None or resource_window.time < end:
                        util: pd.DataFrame = resource_window.resources
                        util = util[util['resource'] == resource]['value'].mean()
                        data['ts'].append(resource_window.time)
                        data['replica_id'].append(fn_replica_id)
                        node = self._replicas[fn_replica_id].node.name
                        data['node'].append(node)
                        data['value'].append(util)
                        data['resource'].append(resource)
            df = pd.DataFrame(data=data)
            return df

    def get_node_cpu(self, node: str, start: int = None, end: int = None) -> Optional[pd.DataFrame]:
        return self.get_node_resource(node, 'cpu', start, end)

    def get_node_resource(self, node: str, resource: str, start: int = None, end: int = None) -> Optional[pd.DataFrame]:
        with self.rw_lock.lock.gen_rlock():
            node_windows = self._node_windows.get(node)
            if node_windows is None:
                return None

            data = defaultdict(list)
            for node_window in node_windows:
                if start is None or node_window.time > start:
                    if end is None or node_window.time < end:
                        value: pd.DataFrame = node_window.resources
                        value = value[value['resource'] == resource]
                        value = value['value'].mean()
                        data['ts'].append(node_window.time)
                        node = self._replicas[node].node.name
                        data['node'].append(node)
                        data['value'].append(value)
                        data['resource'].append(resource)
            return pd.DataFrame(data=data)

