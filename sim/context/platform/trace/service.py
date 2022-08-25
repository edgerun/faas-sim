import pandas as pd
from faas.context import TraceService, InMemoryTraceService
from faas.system import FunctionResponse


class SimTraceService(TraceService):

    def __init__(self, inmemory_trace_service: InMemoryTraceService[FunctionResponse]):
        self.inmemory_trace_service = inmemory_trace_service

    def get_traces_api_gateway(self, node_name: str, start: float, end: float,
                               response_status: int = None) -> pd.DataFrame:
        """contains all traces that were processed in the cluster of the given node"""
        return self.inmemory_trace_service.get_traces_api_gateway(node_name, start, end,
                                                                  response_status)

    def get_traces_for_function(self, function: str, start: float, end: float, zone: str = None,
                                response_status: int = None):
        return self.inmemory_trace_service.get_traces_for_function(function, start, end, zone, response_status)

    def get_traces_for_function_image(self, function: str, function_image: str, start: float, end: float,
                                      zone: str = None, response_status: int = None):
        return self.inmemory_trace_service.get_traces_for_function_image(function, function_image, start, end, zone,
                                                                         response_status)

    def add_trace(self, response: FunctionResponse):
        self.inmemory_trace_service.add_trace(response)
