import abc
from typing import Union, List, Dict, Optional, Callable

from faas.system import FunctionRequest
from faas.util.rwlock import ReadWriteLock


class RequestService(abc.ABC):

    def add_request(self, request: FunctionRequest): ...

    def remove_request(self, request_id: Union[str, int]): ...

    def get_inflight_request(self, node: str) -> List[FunctionRequest]: ...

    def get_end_ts(self, request_id: Union[str, int]) -> Optional[float]: ...


class SimpleRequestService(RequestService):

    def __init__(self, get_time: Callable[[], float]):
        self.rwlock = ReadWriteLock()
        self.get_time = get_time
        # TODO this will leak memory, implement clean up to delete requests that are no relevant anymore
        #      not relevant means that there is no intersection with a running one anymore on the node
        self.requests: Dict[Union[str, int], FunctionRequest] = {}
        self.end_ts: Dict[Union[str, int], float] = {}
        self.start_ts: Dict[Union[str, int], float] = {}

    def add_request(self, request: FunctionRequest):
        with self.rwlock.lock.gen_wlock():
            self.requests[request.request_id] = request
            self.start_ts[request.request_id] = self.get_time()

    def remove_request(self, request_id: Union[str, int]):
        with self.rwlock.lock.gen_wlock():
            self.end_ts[request_id] = self.get_time()

    def get_requests(self, node: str) -> List[FunctionRequest]:
        with self.rwlock.lock.gen_rlock():
            requests = []
            for request in self.requests.values():
                if request.replica.node.name == node:
                    requests.append(request)
            return requests

    def get_end_ts(self, request_id: Union[str, int]) -> Optional[float]:
        with self.rwlock.lock.gen_rlock():
            return self.end_ts.get(request_id)
