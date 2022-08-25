"""
Basically all of this code stems from the jacob-thesis branch.
Thanks, @jjnp for this implementation.
"""
import abc
import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Callable

import numpy as np
from faas.system import FunctionRequest, FunctionReplicaState

from sim.core import Environment
from sim.faas.core import SimFunctionReplica, LocalizedContextLoadBalancer


class WRRProvider(abc.ABC):
    replicas: List[str]

    @abc.abstractmethod
    def next_id(self) -> str:
        pass

    @abc.abstractmethod
    def add_replica(self, replica_id: str):
        pass

    @abc.abstractmethod
    def remove_replica(self, replica_id: str):
        pass

    @staticmethod
    @abc.abstractmethod
    def update_weights(instance: 'WRRProvider', response_times: Dict[str, float]) -> 'WRRProvider':
        pass


class LeastResponseTimeMetricProvider:
    def __init__(self, env: Environment, replica_ids: List[str], window: float = 10.0):
        self.env = env
        self.window = window
        self.replicas = replica_ids
        self.rts = dict()
        self.last_record_timestamps = dict()
        self._init_values()

    def add_replica(self, replica_id: str):
        self.replicas.append(replica_id)
        self.last_record_timestamps[replica_id] = -1
        if len(self.rts.values()) > 0:
            self.rts[replica_id] = float(statistics.median(list(self.rts.values())))
        else:
            self.rts[replica_id] = 0.05

    def remove_replica(self, replica_id: str):
        self.replicas.remove(replica_id)
        del self.rts[replica_id]
        del self.last_record_timestamps[replica_id]

    def _init_values(self):
        for r in self.replicas:
            self.rts[r] = 0.05
            self.last_record_timestamps[r] = -1

    def record_response_time(self, replica_id: str, response_time: float):
        if replica_id not in self.replicas:
            return
        last_record_timestamp = self.last_record_timestamps[replica_id]
        if last_record_timestamp == -1:
            self.last_record_timestamps[replica_id] = self.env.now
            self.rts[replica_id] = response_time
            return
        time_delta = self.env.now - last_record_timestamp
        alpha = 1.0 - math.exp(-time_delta / self.window)
        next_avg_rt = (alpha * response_time) + ((1.0 - alpha) * self.rts[replica_id])
        self.last_record_timestamps[replica_id] = self.env.now
        self.rts[replica_id] = next_avg_rt

    def get_response_times(self) -> Dict[str, float]:
        return self.rts


class SmoothWeightedRoundRobinProvider(WRRProvider):
    def __init__(self, response_times: Dict[str, float], scaling: float = 2.5, max_weight: float = 100):
        self.current_values: Dict[str, float] = defaultdict(lambda: 0)
        self.weights: Dict[str, float] = defaultdict(lambda: 1)
        self.scaling = scaling
        self.max_weight = max_weight
        self.response_times = response_times
        self.weight_sum = 0
        self.replicas = list(response_times.keys())
        if len(response_times) > 0:
            for node_id in response_times.keys():
                self.current_values[node_id] = 0
                self.weights[node_id] = 0
            self.update_weights(self, response_times)

    def add_replica(self, replica_id: str):
        if len(self.current_values) < 1:
            self.current_values[replica_id] = 0
            self.weights[replica_id] = 10.0
            self.weight_sum = 10
            self.replicas.append(replica_id)
            return
        self.current_values[replica_id] = 0
        # self.current_values[replica_id] = float(np.mean(list(self.current_values.values())))
        self.weights[replica_id] = float(np.mean(list(self.weights.values())))
        self.weight_sum += self.weights[replica_id]
        self.replicas.append(replica_id)

    def remove_replica(self, replica_id: str):
        self.weight_sum -= self.weights[replica_id]
        del self.current_values[replica_id]
        del self.weights[replica_id]
        self.replicas = list(self.weights.keys())

    @staticmethod
    def update_weights(instance: 'SmoothWeightedRoundRobinProvider', response_times: Dict[str, float]):
        instance.response_times = response_times
        min_response_time = float(np.min(list(instance.response_times.values())))
        instance.weight_sum = 0
        for node_id, rt in instance.response_times.items():
            weight = float(np.max([1, instance.max_weight / math.pow((rt / min_response_time), instance.scaling)]))
            instance.weights[node_id] = weight
            instance.weight_sum += weight
            instance.current_values[node_id] = 0
        return instance

    def next_id(self) -> str:
        for node_id, weight in self.weights.items():
            self.current_values[node_id] += weight
        chosen_id = max(self.current_values, key=self.current_values.get)
        self.current_values[chosen_id] -= self.weight_sum
        return chosen_id


class DefaultWRRProvider(WRRProvider):
    def __init__(self, response_times: Dict[str, float], scaling: float = 1.0):
        self.gcd = 1
        self.scaling = scaling
        self.replicas = list(response_times.keys())
        random.shuffle(self.replicas)
        self.weights = dict()
        self.cw = 0
        self.last = -1
        self.n = len(response_times)
        self.max_weight = 1
        self.hit_list = {}
        self._set_weights(response_times)
        # if len(self.replicas) > 1:
        #     print(f'WRR count: {len(self.replicas)}')
        # for debugging only

    def __str__(self):
        return str(self.weights)

    def add_replica(self, replica_id: str):

        # use the median weight for the added replica. no special reason
        if len(self.weights.values()) > 0:
            w = int(round(statistics.median(list(self.weights.values()))))
        else:
            w = 10
        self.weights[replica_id] = w
        self.replicas.append(replica_id)
        self.n += 1
        self.gcd = self._calculate_gcd()
        self.hit_list[replica_id] = False

    def remove_replica(self, replica_id: str):
        self.replicas.remove(replica_id)
        del self.weights[replica_id]
        del self.hit_list[replica_id]
        if self.last >= len(self.replicas):
            self.last = -1
        self.n -= 1
        self.gcd = self._calculate_gcd()

    def _set_weights(self, response_times: Dict[str, float]):
        if len(response_times) < 1:
            return
        min_weight = min(response_times.values())
        for r_id, rt in response_times.items():
            w = int(round(max(1.0, pow(10 / (rt / min_weight), self.scaling))))
            self.weights[r_id] = w
            # print('Setting ' + str(r_id) + ' to W' + str(w) + ' based on rt-avg: ' + str(rt))
            self.hit_list[r_id] = False
        self.max_weight = max(self.weights.values())
        self.gcd = self._calculate_gcd()

    def _calculate_gcd(self) -> int:
        weights = self.weights.values()
        max_gcd = min(weights)
        gcd = 1
        for i in range(max_gcd, 0, -1):
            valid = True
            for w in weights:
                if w % i != 0:
                    valid = False
                    break
            if valid and i > 1:
                gcd = i
                break
        # print(f'GCD calclated is: {gcd}')
        return gcd

    def next_id(self) -> str:
        while True:
            self.last = (self.last + 1) % self.n
            if self.last == 0:
                self.cw -= self.gcd
                if self.cw <= 0:
                    self.cw = self.max_weight
            if self.weights[self.replicas[self.last]] >= self.cw:
                self.hit_list[self.replicas[self.last]] = True  # for debugging only
                return self.replicas[self.last]

    @staticmethod
    def update_weights(instance: 'DefaultWRRProvider', response_times: Dict[str, float]):
        return DefaultWRRProvider(response_times)


class LeastResponseTimeLoadBalancer(LocalizedContextLoadBalancer):
    # TODOs
    # [x] Check for new functions and new replicas + integrate them
    # [x] Integrate new replicas in a smarter way (currently we just reset the metrics provider)
    # [x] pay attention to node state (running, etc.)

    def __init__(self, env: Environment, cluster: str,
                 lrt_window: float = 45, weight_update_frequency: float = 30) -> None:
        super().__init__(env, cluster)
        self.count = Counter()
        self.create_wrr: Callable[[Dict[str, float]], WRRProvider] = lambda rts: SmoothWeightedRoundRobinProvider(rts)
        # lrt things
        self.window = lrt_window
        self.weight_update_frequency = weight_update_frequency
        self.last_weight_update = -1
        self.wrr_providers: Dict[str, WRRProvider] = dict()
        self.lrt_providers: Dict[str, LeastResponseTimeMetricProvider] = dict()
        self.hit_list: Dict[str, float] = {}
        self._init_lrt_components()

    # def next_replica(self, request) -> Optional[SimFunctionReplica]:
    #     managed_replicas = self._get_managed_replicas(request)
    #     if len(managed_replicas) == 0:
    #         return None
    #     return random.choice(managed_replicas)
    def next_replica(self, request: FunctionRequest) -> Optional[SimFunctionReplica]:
        managed_replicas = self._get_managed_replicas(request.name)
        if len(managed_replicas) == 0:
            return None
        self._sync_replica_state()
        return self._replica_by_id(request.name, self.wrr_providers[request.name].next_id())

    # def _sync_replica_state(self):
    #     for function_name, replicas in self.replicas.items():
    #         if function_name not in self.wrr_providers.keys():
    #             replica_ids = self.get_running_replica_ids(replicas)
    #             # replica_ids = list(map(lambda r: id(r), replicas))
    #             self.lrt_providers[function_name] = \
    #                 LeastResponseTimeMetricProvider(self.env, replica_ids, window=self.window)
    #             initial_response_times = self.lrt_providers[function_name].get_response_times()
    #             self.wrr_providers[function_name] = WeightedRoundRobinProvider(initial_response_times)
    #         elif len(self.get_running_replica_ids(replicas)) != len(self.lrt_providers[function_name].replicas):
    #             replica_ids = self.get_running_replica_ids(replicas)
    #             self.lrt_providers[function_name] = \
    #                 LeastResponseTimeMetricProvider(self.env, replica_ids, window=self.window)
    #             initial_response_times = self.lrt_providers[function_name].get_response_times()
    #             self.wrr_providers[function_name] = WeightedRoundRobinProvider(initial_response_times)
    def _sync_replica_state(self):
        for function_name in self._get_managed_functions():
            replicas = self._get_managed_replicas(function_name)
            # Entirely new function deployment
            if function_name not in self.wrr_providers.keys():
                replica_ids = self.get_running_replica_ids(replicas)
                for rid in replica_ids:
                    self.hit_list[rid] = False
                self.lrt_providers[function_name] = \
                    LeastResponseTimeMetricProvider(self.env, replica_ids, window=self.window)
                initial_response_times = self.lrt_providers[function_name].get_response_times()
                self.wrr_providers[function_name] = self.create_wrr(initial_response_times)
            current_replica_ids = self.get_running_replica_ids(replicas)
            # replica set of existing function changed
            if not set(current_replica_ids) == set(self.wrr_providers[function_name].replicas):
                for r_id in current_replica_ids:
                    # A new replica as added
                    if r_id not in self.wrr_providers[function_name].replicas:
                        self._add_replica(function_name, r_id)
                for r_id in self.wrr_providers[function_name].replicas:
                    # A replica was removed
                    if r_id not in current_replica_ids:
                        self._remove_replica(function_name, r_id)

    def get_running_replica_ids(self, replicas: List[SimFunctionReplica]):
        replica_ids = [r.replica_id for r in replicas if r.state == FunctionReplicaState.RUNNING]
        return replica_ids

    def _add_replica(self, function_name: str, replica_id: str):
        # print(f'adding replica: {function_name} {self._replica_by_id(replica_id).node.name}')
        self.lrt_providers[function_name].add_replica(replica_id)
        self.wrr_providers[function_name].add_replica(replica_id)
        self.hit_list[replica_id] = False

    def _remove_replica(self, function_name: str, replica_id: str):
        # print('removing replica')
        self.lrt_providers[function_name].remove_replica(replica_id)
        self.wrr_providers[function_name].remove_replica(replica_id)
        if replica_id in self.hit_list.keys():
            del self.hit_list[replica_id]

    def _replica_by_id(self, function: str, replica_id: str) -> Optional[SimFunctionReplica]:
        for r in self._get_managed_replicas(function):
            if r.replica_id == replica_id:
                return r
        return None

    def _get_managed_replicas(self, function: str) -> List[SimFunctionReplica]:
        return self.get_running_replicas(function)

    def _init_lrt_components(self):
        for function_name in self._get_managed_functions():
            replicas = self._get_managed_replicas(function_name)
            replica_ids = self.get_running_replica_ids(replicas)
            self.lrt_providers[function_name] = \
                LeastResponseTimeMetricProvider(self.env, replica_ids, window=self.window)
            initial_response_times = self.lrt_providers[function_name].get_response_times()
            self.wrr_providers[function_name] = self.create_wrr(initial_response_times)
            # self.wrr_providers[function_name] = WeightedRoundRobinProvider(initial_response_times)

    def _should_update_weights(self):
        return self.env.now - self.last_weight_update >= self.weight_update_frequency

    def _get_managed_functions(self) -> List[str]:
        return self.get_functions()

    def _update_weights(self):
        # print('*********************************************')
        # print('Updating weights for LB: ' + str(id(self)))
        for function_name in self._get_managed_functions():
            # Debugging: log out hitlist
            # if len(list(self.wrr_providers[function_name].hit_list.keys())) > 12:
            #     non_hit = []
            #     weight_list = []
            #     for rid, val in self.wrr_providers[function_name].hit_list.items():
            #         if not val:
            #             non_hit.append(self._replica_by_id(rid).node.name)
            #         weight_list.append((self._replica_by_id(rid).node.name, self.wrr_providers[function_name].weights[rid]))
            #     if len(non_hit) > 0:
            #         print(f'{function_name}: {str(weight_list)}')
            #         print(f'{function_name} {str(non_hit)}')
            #         print(self.wrr_providers[function_name].gcd)
            #
            #         print('------------------')
            # print(f'{function_name}: {str(non_hit)}')

            # for r_id, hit in self.wrr_providers[function_name].hit_list.items():
            #     if hit:
            #         self.hit_list[r_id] = True

            response_times = self.lrt_providers[function_name].get_response_times()
            current_wrr = self.wrr_providers[function_name]
            self.wrr_providers[function_name] = current_wrr.update_weights(current_wrr, response_times)

            # w_dict = dict()
            # for r_id, w in self.wrr_providers[function_name].weights.items():
            #     w_dict[self._replica_by_id(r_id).node.ether_node.name] = w
            # print(function_name)
            # print(w_dict)
            # print('--------------------------------------')
        self.last_weight_update = self.env.now
        # frac = len([x for x in self.hit_list.values() if x]) / len(self.hit_list)
        # self.env.metrics.log_load_balancer_hit_fraction(str(id(self)), frac)
        # print('*********************************************')

    def report_response_time(self, request: FunctionRequest, replica: SimFunctionReplica, response_time: float):
        # print('getting response time update: ' + str(response_time))
        self._sync_replica_state()
        self.lrt_providers[request.name].record_response_time(replica.replica_id, response_time)
        if self._should_update_weights():
            # self._sync_replica_state()
            self._update_weights()
            # todo there seems to be a bug here where a replica gets accessed by id even though it doesn't exist (anymore)
            # not sure how this can be with _update_weights, but figure out what's going on...
