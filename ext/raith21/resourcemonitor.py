from typing import Dict, List

import numpy as np

from ext.raith21.functionsim import FunctionCall
from sim.core import Environment
from sim.faas import FaasSystem, FunctionState
from sim.oracle.oracle import ResourceOracle
from sim.resource import ResourceWindow


# TODO migrate this to faas (needs to extract FunctionCall)

class Raith21ResourceMonitor:
    """Simpy process - continuously collects resource data"""

    def __init__(self, env: Environment, resource_oracle: ResourceOracle):
        self.env = env
        self.resource_oracle = resource_oracle
        self.metric_server = env.metrics_server

    def run(self):
        faas: FaasSystem = self.env.faas
        while True:
            start_ts = self.env.now
            yield self.env.timeout(1)
            end_ts = self.env.now
            # calculate resources over function replica resources and save in metric_server
            call_cache: Dict[str, List[FunctionCall]] = {}
            for function_deployment in faas.get_deployments():
                for replica in faas.get_replicas(function_deployment.name, FunctionState.RUNNING):
                    node_name = replica.node.name
                    calls = call_cache.get(node_name, None)
                    if calls is None:
                        calls = replica.node.get_calls_in_timeframe(start_ts, end_ts)
                        call_cache[node_name] = calls
                    trace_execution_durations = []
                    replica_usage = self.resource_oracle.get_resources(node_name, replica.function.image)
                    for call in calls:
                        if call.replica.pod.name == replica.pod.name:
                            last_start = start_ts if start_ts >= call.start else call.start

                            if call.end is not None:
                                first_end = end_ts if end_ts <= call.end else call.end
                            else:
                                first_end = end_ts

                            overlap = first_end - last_start
                            trace_execution_durations.append(overlap)
                    if len(calls) == 0:
                        window = ResourceWindow(replica, 0)
                    else:
                        # TODO because in real life cpu time decreases per function call, thus avoiding the impossible
                        # of getting a cpu time/usage > (cores * second)/100% util, we have to cap this manually
                        # update: tests with PythonHTTPSim have shown that resources (workers=4) effectively can cap this
                        # though it's probably a good safety measure to prevent values > 1 with other simulators
                        sum = np.sum(trace_execution_durations)
                        # this should be enough, because: our time window is 1 second, if one call consumed 100%
                        # -> which means that *all* cores were running 100% of the time, thus the final
                        cpu_usage = (sum * replica_usage['cpu'])
                        # cpu = cpu_usage / int(replica.node.ether_node.capacity.cpu_millis / 1000)
                        window = ResourceWindow(replica, min(1, cpu_usage))
                    self.metric_server.put(window)
