import logging
from collections import defaultdict
from typing import Callable, Optional, Dict, List

import pandas as pd
from faas.context import TraceService, NodeService, ResponseRepresentation
from faas.system import FunctionResponse
from faas.util.point import PointWindow, Point
from faas.util.rwlock import ReadWriteLock

from sim.context.platform.node.model import SimFunctionNode

logger = logging.getLogger(__name__)


class SimTraceService(TraceService):

    """
    This implementation keeps a dict, whereas the key is the request id, and the list is all traces with this id
    then, if someone quueries for a function, get all request_ids associated with this function/function-image
    and fetch for each request_id, from the request_id dict, the list and get the request with the highest exec time.
    then, set modify the requests, such that the client, replica node as well as function and function image correspond
    to that, what the callee asked for
    """

    def __init__(self,now: Callable[[],float], window_size: int, node_service: NodeService,
                 parser: Callable[[FunctionResponse], Optional[ResponseRepresentation]]):
        self.now = now
        self.window_size = window_size
        self.node_service = node_service
        self.parser = parser
        # collect all requests by id, when using a load balancer one call might result in multiple responses
        self.requests_by_id: Dict[int, List[ResponseRepresentation]] = defaultdict(list)
        self.requests_per_node: Dict[str, PointWindow[ResponseRepresentation]] = {}
        self.locks = {}
        self.last_purge = 0
        self.request_by_id_lock = ReadWriteLock()
        self.request_cache = {}
        # TODO does not support new nodes during experiments
        for node in node_service.get_nodes():
            self.locks[node.name] = ReadWriteLock()

    def _purge(self, till_ts: float):
        for node, point_window in self.requests_per_node.items():
            with self.locks[node].lock.gen_wlock():
                purged  = point_window.purge(till_ts)
                for purge in purged:
                    purge_id = purge.val.request_id
                    try:
                        del self.requests_by_id[purge_id]
                        del self.request_cache[purge_id]
                    except KeyError:
                        pass

    def purge(self):
        now = self.now()
        duration_since_last_purge = now - self.last_purge
        if duration_since_last_purge > self.window_size:
            self._purge(now - self.window_size)
            self.last_purge = now

    def get_traces_api_gateway(self, node_name: str, start: float, end: float,
                               response_status: int = None) -> pd.DataFrame:
        self.purge()
        gateway = self.node_service.find(node_name)
        if gateway is None:
            nodes = self.node_service.get_nodes_by_name()
            raise ValueError(f"Node {node_name} not found, currently stored: {nodes}")
        zone = gateway.cluster
        nodes = self.node_service.find_nodes_in_zone(zone)
        if len(nodes) == 0:
            logger.info(f'No nodes found in zone {zone}')
        request_ids = self._get_request_ids_for_nodes(nodes, start, end, response_status)
        requests = self._get_requests(request_ids)
        return requests

    def add_trace(self, response: FunctionResponse):
        with self.locks[response.node.name].lock.gen_wlock():
            node = response.node.name
            window = self.requests_per_node.get(node, None)
            if window is None:
                self.requests_per_node[node] = PointWindow(self.window_size)
            self.requests_per_node[node].append(Point(response.request.start, self.parser(response)))

        with self.request_by_id_lock.lock.gen_wlock():
            self.requests_by_id[response.request_id].append(self.parser(response))

    def _get_request_ids_for_nodes(self, nodes: List[SimFunctionNode], start: float, end: float, response_status:int=None) -> List[int]:
        requests = defaultdict(list)

        for node in nodes:
            with self.locks[node.name].lock.gen_rlock():
                node_requests = self.requests_per_node.get(node.name)
                if node_requests is None or node_requests.size() == 0:
                    continue

                for req in node_requests.value():
                    for key, value in req.get_value().__dict__.items():
                        requests[key].append(value)
        df = pd.DataFrame(data=requests).sort_values(by='ts')
        df.index = pd.DatetimeIndex(pd.to_datetime(df['ts'], unit='s'))

        df = df[df['ts'] >= start]
        df = df[df['ts'] <= end]
        if response_status is not None:
            df = df[df['status'] == response_status]
            logger.info(f'After filtering out non status: {len(df)}')
        return df['request_id'].unique().tolist()

    def _get_request_ids(self, start: float, end: float, zone: str = None,
                                response_status: int = None, function_name: str=None, function_image:str=None) -> List[int]:
        if zone is not None:
            nodes = self.node_service.find_nodes_in_zone(zone)
        else:
            nodes = self.node_service.get_nodes()
        requests = set()
        for node in nodes:
            node_name = node.name
            with self.locks[node_name].lock.gen_rlock():
                node_requests = self.requests_per_node.get(node_name)
                if node_requests is None or node_requests.size() == 0:
                    continue

                for req in node_requests.value():
                    if function_name is not None and req.val.function != function_name:
                        continue
                    if req.val.ts >= start or req.val.ts <= end:
                        if function_image is None or req.val.function_image == function_image:
                            if response_status is None or req.val.status == response_status:
                                requests.add(req.val.request_id)
        return list(requests)

    get_values_function_cache = {}

    def get_values_for_function(self, function: str, start: float, end: float, access: Callable[[ResponseRepresentation], List[float]],
                                zone: str = None, response_status: int = None):
        request_ids = self._get_request_ids(start, end, zone, response_status, function)
        with self.request_by_id_lock.lock.gen_rlock():
            request_data = []

            for request_id in request_ids:
                if self.request_cache.get(request_id) is not None:
                    representation = self.request_cache[request_id]
                    request_data.append(access(representation))
                else:
                    requests = self.requests_by_id[request_id]
                    max_rtt = 0
                    max_response = None
                    last_sent = 0
                    last_response = None
                    for request in requests:
                        if request.rtt > max_rtt:
                            # this is the invocation of the client to load balancer
                            max_rtt = request.rtt
                            max_response = request
                        if request.sent > last_sent:
                            # this is the last invocation from load balancer to actual replica
                            last_response = request
                            last_sent = request.sent

                    representation = ResponseRepresentation(
                        ts=max_response.ts,
                        function=last_response.function,
                        function_image=last_response.function_image,
                        replica_id=last_response.replica_id,
                        node=last_response.node,
                        rtt=max_response.rtt,
                        done=max_response.done,
                        sent=max_response.sent,
                        origin_zone=max_response.origin_zone,
                        dest_zone=last_response.dest_zone,
                        client=max_response.client,
                        status=max_response.status,
                        request_id=request_id
                    )

                    request_data.append(access(representation))
                    self.request_cache[request_id] = representation

        return request_data

    def _get_requests(self, request_ids: List[int]) -> Optional[pd.DataFrame]:
        with self.request_by_id_lock.lock.gen_rlock():
            request_data = defaultdict(list)

            for request_id in request_ids:
                if self.request_cache.get(request_id) is not None:
                    representation = self.request_cache[request_id]
                    for key, value in representation.__dict__.items():
                        request_data[key].append(value)
                else:
                    requests = self.requests_by_id[request_id]
                    max_rtt = 0
                    max_response = None
                    last_sent = 0
                    last_response = None
                    for request in requests:
                        if request.rtt > max_rtt:
                            # this is the invocation of the client to load balancer
                            max_rtt = request.rtt
                            max_response = request
                        if request.sent > last_sent:
                            # this is the last invocation from load balancer to actual replica
                            last_response = request
                            last_sent = request.sent

                    representation = ResponseRepresentation(
                        ts=max_response.ts,
                        function=last_response.function,
                        function_image=last_response.function_image,
                        replica_id=last_response.replica_id,
                        node=last_response.node,
                        rtt=max_response.rtt,
                        done=max_response.done,
                        sent=max_response.sent,
                        origin_zone=max_response.origin_zone,
                        dest_zone=last_response.dest_zone,
                        client=max_response.client,
                        status=max_response.status,
                        request_id=request_id
                    )

                    for key, value in representation.__dict__.items():
                        request_data[key].append(value)
                    self.request_cache[request_id] = representation

            df = pd.DataFrame(data=request_data)
            if len(df) == 0:
                return None
            df = df.sort_values(by='ts')
            df.index = pd.DatetimeIndex(pd.to_datetime(df['ts'], unit='s'))
            df = df.reset_index(drop=True)
            return df

    def get_traces_for_function(self, function_name: str, start: float, end: float, zone: str = None,
                                response_status: int = None) -> Optional[pd.DataFrame]:
        self.purge()

        request_ids = self._get_request_ids(start,end,zone,response_status, function_name=function_name)
        request_df = self._get_requests(request_ids)
        return request_df


    def get_traces_for_function_image(self, function: str, function_image: str, start: float, end: float,
                                      zone: str = None,
                                      response_status: int = None) -> Optional[pd.DataFrame]:
        self.purge()

        request_ids = self._get_request_ids(start, end, zone, response_status, function, function_image)
        request_df = self._get_requests(request_ids)
        return request_df
