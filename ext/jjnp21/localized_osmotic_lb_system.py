from collections import defaultdict
from typing import Dict, List, Set

from ext.jjnp21.automator.factories.lb_scaler import LoadBalancerScalerFactory
from ext.jjnp21.localized_lb_system import LocalizedLoadBalancerFaasSystem
from sim.core import Environment
from sim.faas import FunctionRequest
from sim.faas.core import Node
import numpy as np

class PingCache:
    def __init__(self, env: Environment):
        self.env = env
        self.cache: Dict[(Node, Node), float] = {}

    def get_distance(self, first_node_name: Node, second_node_name: Node) -> float:
        if first_node_name == second_node_name:
            return 0
        if (first_node_name, second_node_name) in self.cache:
            return self.cache[(first_node_name, second_node_name)]
        if (second_node_name, first_node_name) in self.cache:
            return self.cache[(second_node_name, first_node_name)]
        # we check both directions since it doesn't make a difference to the result
        self.cache[(first_node_name, second_node_name)] = self.env.topology.latency(first_node_name, second_node_name)
        return self.cache[(first_node_name, second_node_name)]

class OsmoticLoadBalancerCapableFaasSystem(LocalizedLoadBalancerFaasSystem):
    def __init__(self, env: Environment, lb_scaler_factory: LoadBalancerScalerFactory,
                 scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False,
                 scale_static: bool = False, lb_osmotic_pressure_threshold: float = 0.3,
                 lb_osmotic_pressure_hysteresis: float = 0.1,
                 lb_osmotic_pressure_window_size: float = 60):
        super().__init__(env, lb_scaler_factory, scale_by_requests, scale_by_average_requests,
                         scale_by_queue_requests_per_replica, scale_static)
        self.lb_osmotic_pressure_threshold = lb_osmotic_pressure_hysteresis #todo shouldn't this go to the scaler?
        self.lb_osmotic_pressure_hysteresis = lb_osmotic_pressure_hysteresis
        self.lb_osmotic_pressure_window_size = lb_osmotic_pressure_window_size

        self.distance_cache = PingCache(self.env)

        # dict that stores sent requests: Dict[function_name, Dict[client_node_name, List[request sent time]]]
        self.request_log: Dict[str, Dict[Node, List[float]]] = defaultdict(lambda: defaultdict(lambda: []))

    def calculate_pressures(self) -> Dict[Node, float]:
        pass
        """
        TODO: get clients that would be attached to node if it were a LB
         - for each client get the LB
         - if distance client-LB > than client-node, then node gets the client
         - get the request share the client would have per function
         - get the list of replicas for each function
         - calculate the distance from the replicas and weigh it appropriately
        """
        # all nodes. client nodes are not included, docker registry is not included
        all_nodes = self.env.cluster.list_nodes()
        pressures: Dict[Node, float] = {}
        # nodes with currently running load balancers
        lb_nodes = set()
        for replica_list in self.lb_replicas.values():
            for replica in replica_list:
                lb_nodes.add(replica.node.ether_node)

        interval_start = max([0, self.env.now - self.lb_osmotic_pressure_window_size])

        rq_log_in_interval: Dict[str, Dict[Node, List[float]]] = defaultdict(lambda: defaultdict(lambda: []))
        for fn_name, logs in self.request_log.items():
            for node, requests in logs.items():
                requests_in_interval = [r for r in requests if r >= interval_start]
                rq_log_in_interval[fn_name][node] = requests_in_interval

        fn_totals: Dict[str, int] = {}
        for fn_name, logs in rq_log_in_interval.items():
            fn_totals[fn_name] = sum([len(rqs) for rqs in logs.values()])
        for node in all_nodes:
            pressures[node] = self._calc_p(node, rq_log_in_interval, fn_totals)
        return pressures



    def _calc_p(self, node: Node, rq_log: Dict[str, Dict[Node, List[float]]], fn_totals: Dict[str, int]):
        function_replica_closeness_impact_factor = 0.1 #todo IMPORTANT! re-evaluate that one!

        clients = self._get_potential_clients(node)
        request_shares = self._calc_request_shares(clients, rq_log)
        fn_distances_sorted = self._get_fx_distances_sorted(node)
        fn_closeness_factors: Dict[str, float] = {}
        for fn_name, distances in fn_distances_sorted.items():
            rq_share = request_shares[fn_name]
            # currently the fn distance is defined as:
            # the mean distance of the s% closest function replicas, where
            # s = the share the node has on total requests
            # i.e. the more load the node would have, the more function replicas we consider for distance
            fn_closeness_factors[fn_name] = float(np.mean([d for d in distances[:int(len(distances) * rq_share)]]))

        fn_pressures: Dict[str, float] = {}
        total_request_count = sum(fn_totals.values())
        for fn_name, fn_distance in fn_closeness_factors.items():
            # todo: figure out the scaling between these two
            # also: 1 / fn_distance has no upper limit since 1 / 0.00000001 is very large etc.
            # also: there is a division by 0 error waiting to happen on co-located nodes.
            # todo: check if latencies are measured in seconds or milliseconds

            # idea: <how important is the function> * <how much load would the lb get> + <function closeness> * <how important is function closeness>
            fn_pressures[fn_name] = (fn_totals[fn_name] / total_request_count) * request_shares[fn_name] + (1 / fn_distance) * function_replica_closeness_impact_factor
        return sum(fn_pressures.values())



    def _get_fx_distances_sorted(self, node: Node) -> Dict[str, List[float]]:
        distances: Dict[str, List[float]] = defaultdict(lambda: [])
        for fn_name in self.replicas.keys():
            fn_nodes = [r.node.ether_node for r in self.get_replicas(fn_name)]
            distances[fn_name].extend([self.distance_cache.get_distance(node, r_node) for r_node in fn_nodes])
            # sort() will sort ascending, i.e. closest nodes first
            distances[fn_name].sort()
        return distances

    def _calc_request_shares(self, clients: List[Node], rq_log: Dict[str, Dict[Node, List[float]]]) -> Dict[str, float]:
        shares: Dict[str, float] = {}
        for fn_name in self.replicas.keys():
            total = sum([len(rqs) for rqs in rq_log[fn_name].values()])
            client_sum = sum([len(rq_log[fn_name][c]) for c in clients])
            shares[fn_name] = client_sum / total
        return shares


    def _get_potential_clients(self, node: Node) -> List[Node]:
        clients = []
        for c in self.client_nodes:
            lb_node = self.lb_finder.get_closest_lb(c).node.ether_node
            if self.distance_cache.get_distance(c, lb_node) >= self.distance_cache.get_distance(c, node):
                clients.append(c)
        return clients


    def set_request_client(self, request: FunctionRequest):
        # Note: request.name denotes the FUNCTION the request wants. Yes this is not intuitive, but it was so when I started
        # and I'm not touching it
        super().set_request_client(request)
        self.request_log[request.name][request.client_node].append(self.env.now)


    def osmotic_scale_up_lb(self, lb_name: str, target_node: Node):
        # todo: create pod and everything with labels according to the params
        pass

    def osmotic_scale_down_lb(self, lb_name: str, target_node: str):
        # todo work out how to scale down on low enough pressures
        pass

    def scale_up_lb(self, lb_name: str, add_count: int):
        return super().scale_up_lb(lb_name, add_count)

    def scale_down_lb(self, lb_name: str, remove_count: int):
        super().scale_down_lb(lb_name, remove_count)
