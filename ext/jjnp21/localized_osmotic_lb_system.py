import copy
import logging
import statistics
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np

from ext.jjnp21.automator.factories.lb_scaler import LoadBalancerScalerFactory
from ext.jjnp21.core import LoadBalancerDeployment, LoadBalancerReplica
from ext.jjnp21.load_balancers.lrt import LeastResponseTimeLoadBalancer
from ext.jjnp21.localized_lb_system import LocalizedLoadBalancerFaasSystem, NetworkSimulationMode
from ext.jjnp21.topology import client_label
from sim.core import Environment
from sim.faas import FunctionRequest
from sim.faas.core import Node, FunctionContainer, FunctionSimulator, FunctionState
from sim.topology import DockerRegistry

logger = logging.getLogger(__name__)


class PingCache:
    def __init__(self, env: Environment):
        self.env = env
        self.cache: Dict[(Node, Node), float] = {}

    def _distance(self, source: Node, sink: Node):
        samples = []
        for _ in range(10):
            samples.append(self.env.topology.latency(source, sink))
        return statistics.mean(samples)

    def get_distance(self, first_node_name: Node, second_node_name: Node) -> float:
        # Note: returns response time in ms. i.e. 5ms = 5.0 (and NOT 0.005 like it would be in full seconds)
        if first_node_name == second_node_name:
            return 0
        if (first_node_name, second_node_name) in self.cache:
            return self.cache[(first_node_name, second_node_name)]
        if (second_node_name, first_node_name) in self.cache:
            return self.cache[(second_node_name, first_node_name)]
        # we check both directions since it doesn't make a difference to the result
        # self.cache[(first_node_name, second_node_name)] = self._distance(first_node_name, second_node_name)
        self.cache[(first_node_name, second_node_name)] = self.env.topology.latency(first_node_name, second_node_name)
        return self.cache[(first_node_name, second_node_name)]


class OsmoticLoadBalancerCapableFaasSystem(LocalizedLoadBalancerFaasSystem):
    def __init__(self, env: Environment, lb_scaler_factory: LoadBalancerScalerFactory,
                 scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False,
                 scale_static: bool = False,
                 net_mode: NetworkSimulationMode = NetworkSimulationMode.ACCURATE,
                 lb_osmotic_pressure_window_size: float = 60):
        super().__init__(env, lb_scaler_factory, scale_by_requests, scale_by_average_requests,
                         scale_by_queue_requests_per_replica, scale_static, net_mode=net_mode)
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
        # all_nodes = self.env.cluster.list_nodes()
        all_nodes = [node for node in self.env.topology.get_nodes() if
                     node.labels.get(client_label, None) is None and node != DockerRegistry]
        pressures: Dict[Node, float] = {}
        # nodes with currently running load balancers
        lb_nodes = set()
        for replica_list in self.lb_replicas.values():
            for replica in replica_list:
                if replica.state == FunctionState.RUNNING:
                    lb_nodes.add(replica.node.ether_node)

        interval_start = max([0, self.env.now - self.lb_osmotic_pressure_window_size])

        rq_log_in_interval: Dict[str, Dict[Node, List[float]]] = defaultdict(lambda: defaultdict(lambda: []))
        for fn_name, logs in self.request_log.items():
            for node, requests in logs.items():
                requests_in_interval = [r for r in requests if r >= interval_start]
                rq_log_in_interval[fn_name][node] = requests_in_interval

        # how important is each function based on rps? Dict[fn_name, fn_importance]
        fn_relative_importance: Dict[str, float] = defaultdict(lambda: 0)
        fn_total_request_sum = sum([len(rqs) for node_rqs in rq_log_in_interval.values() for rqs in node_rqs.values()])
        for fn_name, logs in rq_log_in_interval.items():
            fn_relative_importance[fn_name] = sum([len(rqs) for rqs in logs.values()]) / fn_total_request_sum

        # client stuff per node Dict[Node, Dict[fn_name, mean_dist]]
        client_distances: Dict[Node, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
        client_assignments: Dict[Node, List[Node]] = defaultdict(list)
        # request shares per node and fn Dict[Node, Dict[fn_name, rps_share]]
        request_shares: Dict[Node, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0))
        # fn distances Dict[Node, Dict[fn_name, avg_fn_replica_distance]]
        fn_distances: Dict[Node, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 10000))
        fn_sorted_distances: Dict[Node, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        lb_client_count_sum = 0
        # todo add a node list per client for the whole debugging stuff
        debug_lb_client_list = []
        for node in all_nodes:
            clients, distances = self._get_potential_clients(node, node in lb_nodes)
            # todo remove me. just for debugging
            if node in lb_nodes:
                lb_client_count_sum += len(clients)
                debug_lb_client_list.extend(clients)
            client_distances[node] = self._calculate_client_distances(clients, distances, rq_log_in_interval)
            client_assignments[node] = clients
            request_shares[node] = self._calc_request_shares(clients, rq_log_in_interval)
            fn_distances_sorted = self._get_fx_distances_sorted(node)
            fn_sorted_distances[node] = fn_distances_sorted
            for fn_name, rq_share in request_shares[node].items():
                distances = fn_distances_sorted[fn_name]
                distance_end_index = int(np.max([1, int(len(distances) * rq_share)]))
                fn_distances[node][fn_name] = float(
                    np.max([1, float(np.mean([d for d in distances[:distance_end_index]]))]))

        # calculate status quo
        # status quo performance Dict[fn_name, fn_performance]
        status_quo_performance: Dict[str, float] = defaultdict(lambda: 0)
        status_quo_performances: Dict[str, List[float]] = defaultdict(list)
        status_quo_performances_weights: Dict[str, List[float]] = defaultdict(list)
        fn_debug_sums: Dict[str, float] = defaultdict(lambda: 0)
        for lb_node in lb_nodes:
            lb_node_perf = self._calculate_projected_performance(request_shares[lb_node], fn_distances[lb_node],
                                                                 client_distances[lb_node])
            for fn_name, fn_perf in lb_node_perf.items():
                if not np.isnan(fn_perf):
                    status_quo_performances[fn_name].append(fn_perf)
                    status_quo_performances_weights[fn_name].append(request_shares[lb_node][fn_name])
                    status_quo_performance[fn_name] += fn_perf * request_shares[lb_node][fn_name]
                else:
                    pass
                fn_debug_sums[fn_name] += request_shares[lb_node][fn_name]

        #start of test
        test_status_quo: Dict[str, float] = defaultdict(lambda: 0.0000001)
        for function, pfs in status_quo_performances.items():
            # relevant = list(filter(lambda x: x != 0, pfs))
            # if len(relevant) == 0:
            #     test_status_quo[function] = 0.00001
            #     continue
            # test_status_quo[function] = float(np.median(relevant))
            # test_status_quo[function] = float(np.median(relevant))
            a = float(weighted_quantile(pfs, [0.5], status_quo_performances_weights[function])[0])
            if a < 0:
                print('kek')
            test_status_quo[function] = a
            # if test_status_quo[function] == 0:
            #     test_status_quo[function] = 0.000001

        # end of test

        for node in all_nodes:
            if node in lb_nodes:
                pressures[node] = self._calc_neg(node, client_assignments, rq_log_in_interval, fn_sorted_distances, lb_nodes, status_quo_performance, fn_relative_importance)
            else:
                pressures[node] = self._calc_p_new(request_shares[node], fn_distances[node], client_distances[node], fn_relative_importance, test_status_quo, len(lb_nodes), status_quo_performances)
                # pressures[node] = self._calc_p_new(request_shares[node], fn_distances[node], client_distances[node], fn_relative_importance, status_quo_performance, len(lb_nodes), status_quo_performances)

        # simple new test idea:
        # for lb de-scheduling we simply assign the clients to the relatively speaking closest LB and then recalculate performance from there
        # this should generally speaking be pretty fast. Will it be terribly, terribly accurate? No, but it also doesn't require a completely new recalculation
        # The difference this will make mostly depends on how coherent the client nodes currently are. The more coherent the more of a difference it will make

        #todo current status quo:
        # new lb descaling idea hasn't really been evaluated.
        # double assignment on status quo calulation unfortunately remains.
        # potentially status quo has to be it's own whole type of calulation, where other nodes are simply not checked at all.
        # a solution would be that for active LB nodes we simply check the lb_finder records, and nothing else.
        # frankly I need to do "something" and then just stick with it. The sad part is that my solution doesn't even
        # clearly outperform random assignment from what it looks like...

        # just for debugging
        node_pressures = {}
        npc = defaultdict(dict)
        lb_pressures = {}
        lpc = defaultdict(dict)
        for node, pressure in pressures.items():
            if node in lb_nodes:
                lb_pressures[node] = pressure
                lpc[node.labels['city']][node] = pressure
            else:
                node_pressures[node] = pressure
                npc[node.labels['city']][node] = pressure
        # end of debugging

        return pressures



    def _calc_neg(self, node: Node,
                  client_assignments: Dict[Node, List[Node]], rq_log: Dict[str, Dict[Node, List[float]]],
                  fn_sorted_distances: Dict[Node, Dict[str, List[float]]], lb_nodes: Set[Node],
                  status_quo_performance: Dict[str, float], fn_relative_importance: Dict[str, float]):
        if len(lb_nodes) <= 1 or np.nansum(list(status_quo_performance.values())) == 0:
            return 0

        if len(client_assignments[node]) == 0:
            return -1
        # find closest lb_node
        min_dist = 100000
        lb: Node = None
        for lb_node in lb_nodes:
            dist = self.distance_cache.get_distance(node, lb_node)
            if dist < min_dist and lb_node != node:
                min_dist = dist
                lb = lb_node
        joined_clients: List[Node] = []
        joined_clients.extend(client_assignments[lb])  # add original lb clients
        joined_clients.extend(client_assignments[node])  # add new lb clients
        distances: List[float] = []
        for client in joined_clients:
            distances.append(self.distance_cache.get_distance(client, lb))

        lb_request_share = self._calc_request_shares(joined_clients, rq_log)
        lb_client_distances = self._calculate_client_distances(joined_clients, distances, rq_log)
        lb_fn_distances = self._calculate_fx_distances(fn_sorted_distances[lb], lb_request_share)
        lb_perf = self._calculate_projected_performance(lb_request_share, lb_fn_distances, lb_client_distances)

        node_removal_pressure: Dict[str, float] = defaultdict(lambda: 0)
        # todo: instead of this we could calculate the "actual" new performance, no?
        for function, status_quo in status_quo_performance.items():
            if status_quo == 0:
                continue
            projected_performance = (lb_request_share[function] * lb_perf[function]) + ((1 - lb_request_share[function]) * status_quo)
            delta = (status_quo - projected_performance) / status_quo
            # here the calulation is reversed since we want a negative value iff removing (i.e. new perf) is better than status quo
            node_removal_pressure[function] = delta * fn_relative_importance[function]
        return np.nansum(list(node_removal_pressure.values()))

    def _calculate_fx_distances(self, fn_distances_sorted: Dict[str, List[float]], rq_shares: Dict[str, float]) -> Dict[str,float]:
        fn_distances: Dict[str, float] = defaultdict(lambda: 100000)
        for fn_name, rq_share in rq_shares.items():
            distances = fn_distances_sorted[fn_name]
            distance_end_index = int(np.max([1, int(len(distances) * rq_share)]))
            fn_distances[fn_name] = float(np.max([1, float(np.mean([d for d in distances[:distance_end_index]]))]))
        return fn_distances


    def _calculate_projected_performance(self, request_shares: Dict[str, float], fn_distances: Dict[str, float],
                                         client_distances: Dict[str, float]) -> Dict[str, float]:
        perf_per_fn: Dict[str, float] = defaultdict(lambda: 0)
        for fn_name, _ in request_shares.items():
            perf_per_fn[fn_name] = 1 / (fn_distances[fn_name] + client_distances[fn_name])
            # perf_per_fn[fn_name] = ((1 / fn_distances[fn_name]) + (1 / client_distances[fn_name]))
            # perf_per_fn[fn_name] = rq_share * ((1 / fn_distances[fn_name]) + (1 / client_distances[fn_name]))
        return perf_per_fn

    def _calc_p_new(self, request_shares: Dict[str, float], fn_distances: Dict[str, float],
                    client_distances: Dict[str, float], fn_relative_importance: Dict[str, float],
                    status_quo_performance: Dict[str, float], lb_count: int, status_quo_performances: Dict[str, List[float]]) -> float:
        node_perf = self._calculate_projected_performance(request_shares, fn_distances, client_distances)
        fn_pressure: Dict[str, float] = {}
        for fn_name, fn_importance in fn_relative_importance.items():
            projected_performance = (request_shares[fn_name] * node_perf[fn_name]) + (
                    (1 - request_shares[fn_name]) * status_quo_performance[fn_name])
            projected_performance_delta = (projected_performance - status_quo_performance[fn_name]) / \
                                          status_quo_performance[fn_name]
            fn_pressure[fn_name] = projected_performance_delta * fn_importance

            # avg_status_quo_perf = status_quo_performance[fn_name] / lb_count
            # projected_performance = request_shares[fn_name] * node_perf[fn_name]
            # fn_pressure[fn_name] = (projected_performance - avg_status_quo_perf) / avg_status_quo_perf * fn_importance

            # fn_pressure[fn_name] = (node_perf[fn_name] - status_quo_performance[fn_name]) / status_quo_performance[fn_name] * fn_importance

            # fn_pressure[fn_name] = request_shares[fn_name] * (1 / client_distances[fn_name])
            # fn_pressure[fn_name] = request_shares[fn_name] + 0.01 * node_perf[fn_name]


            # if there are no request shares we need to have this penalty factor, as the load balancer is in effect useless
            # delta = ((node_perf[fn_name] * request_shares[fn_name]) - status_quo_performance[fn_name]) / status_quo_performance[fn_name]
            # fn_pressure[fn_name] = delta * fn_importance
            if request_shares[fn_name] == 0:
                fn_pressure[fn_name] = -1 * fn_importance
        pressure = np.nansum(list(fn_pressure.values()))
        return pressure


    def _calc_p(self, node: Node, rq_log: Dict[str, Dict[Node, List[float]]], fn_totals: Dict[str, int]) -> float:
        function_replica_closeness_impact_factor = 0.1  # todo IMPORTANT! re-evaluate that one!

        clients, _ = self._get_potential_clients(node)
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
            if fn_distance == 0:
                fn_distance = 1
            if total_request_count == 0:
                fn_pressures[fn_name] = 0
                continue

            # idea: <how important is the function> * <how much load would the lb get> + <function closeness> * <how important is function closeness>
            # fn_pressures[fn_name] = (fn_totals[fn_name] / total_request_count) * request_shares[fn_name] + (
            #         1 / fn_distance) * function_replica_closeness_impact_factor
            # fn_pressures[fn_name] = (fn_totals[fn_name] / total_request_count) * request_shares[fn_name] * (
            #         1 / fn_distance)
            # fn_pressures[fn_name] = (fn_totals[fn_name] / total_request_count) * request_shares[fn_name] + request_shares[fn_name] * (1 / fn_distance) * 0.1
            fn_pressures[fn_name] = (fn_totals[fn_name] / total_request_count) * request_shares[fn_name]
        return float(np.nansum(list(fn_pressures.values())))

    def _calculate_client_distances(self, clients: List[Node], distances: List[float],
                                    rq_log: Dict[str, Dict[Node, List[float]]]) -> Dict[str, float]:
        distance_sums: Dict[str, float] = defaultdict(lambda: 0)
        request_sums: Dict[str, int] = defaultdict(lambda: 0)
        result_distances: Dict[str, float] = defaultdict(lambda: 10000)
        for i, client in enumerate(clients):
            for fn_name, rqs in rq_log.items():
                request_count = len(rqs[client])
                distance_sums[fn_name] += distances[i] * request_count
                request_sums[fn_name] += request_count
        for fn_name, request_sum in request_sums.items():
            result_distances[fn_name] = float(np.max([1, distance_sums[fn_name] / request_sum]))
        return result_distances

    def _get_fx_distances_sorted(self, node: Node) -> Dict[str, List[float]]:
        distances: Dict[str, List[float]] = defaultdict(lambda: [])
        for fn_name in self.replicas.keys():
            fn_nodes = [r.node.ether_node for r in self.get_replicas(fn_name, state=FunctionState.RUNNING)]
            distances[fn_name].extend([self.distance_cache.get_distance(node, r_node) for r_node in fn_nodes])
            # sort() will sort ascending, i.e. closest nodes first
            distances[fn_name].sort()
        return distances

    def _calc_request_shares(self, clients: List[Node], rq_log: Dict[str, Dict[Node, List[float]]]) -> Dict[str, float]:
        shares: Dict[str, float] = {}
        for fn_name in self.replicas.keys():
            total = sum([len(rqs) for rqs in rq_log[fn_name].values()])
            client_sum = sum([len(rq_log[fn_name][c]) for c in clients])
            if total == 0 or client_sum == 0:
                shares[fn_name] = 0
                continue
            shares[fn_name] = client_sum / total
        return shares

    def _get_potential_clients(self, node: Node, is_load_balancer: bool = False) -> (List[Node], float):
        clients = []
        distances = []
        for c in self.client_nodes:
            lb_node = self.lb_finder.get_closest_lb(c).node.ether_node
            client_to_node_distance = self.distance_cache.get_distance(c, node)
            # idea: maybe return distance from lb-finder as well. This way measurements might be more consistent
            # problem could also be the result of the equals sign, if multiple nodes are "torn" between two lbs.
            client_to_lb_distance = self.distance_cache.get_distance(c, lb_node)
            if is_load_balancer and node.name == lb_node.name:
                clients.append(c)
                distances.append(client_to_node_distance)
            if not is_load_balancer and (client_to_lb_distance >= client_to_node_distance + 1 or client_to_lb_distance == client_to_node_distance):
                clients.append(c)
                distances.append(client_to_node_distance)
        return (clients, distances)

    def set_request_client(self, request: FunctionRequest):
        # Note: request.name denotes the FUNCTION the request wants. Yes this is not intuitive, but it was so when I started
        # and I'm not touching it
        super().set_request_client(request)
        self.request_log[request.name][request.client_node].append(self.env.now)

    def osmotic_scale_up_lb(self, lb_name: str, target_nodes: List[Node]):
        # todo: create pod and everything with labels according to the params
        ld = self.lb_deployments[lb_name]

        svc = ld.get_services()[0]

        for target_node in target_nodes:
            yield from self.osmotic_deploy_lb_replica(ld, ld.get_container(svc.image), ld.get_containers(), target_node)

    def osmotic_deploy_lb_replica(self, ld: LoadBalancerDeployment, fn: FunctionContainer,
                                  services: List[FunctionContainer], target_node: Node):
        replica = self.osmotic_create_lb_replica(ld, fn, target_node)
        self.lb_replicas[ld.name].append(replica)
        self.env.metrics.log_queue_schedule(replica)
        self.env.metrics.log_function_replica(replica)
        yield self.lb_scheduler_queue.put((replica, services))

    def get_closest_lb_replica_instance(self, node: Node, ld: LoadBalancerDeployment) -> (LoadBalancerReplica, float):
        min_dist = 10000000000
        min_replica = None
        for replica in self.lb_replicas[ld.name]:
            dist = self.distance_cache.get_distance(node, replica.node.ether_node)
            if dist < min_dist:
                min_dist = dist
                min_replica = replica
        return min_replica, min_dist

    def osmotic_create_lb_replica(self, ld: LoadBalancerDeployment, fn: FunctionContainer,
                                  target_node: Node) -> LoadBalancerReplica:
        replica = LoadBalancerReplica()
        replica.function = ld
        replica.container = fn
        replica.load_balancer = ld.create_load_balancer(self.env, self.replicas)

        # todo add initial load balancing values here.
        #
        closest_replica, min_dist = self.get_closest_lb_replica_instance(target_node, ld)
        if min_dist < 50:
            if isinstance(closest_replica.load_balancer, LeastResponseTimeLoadBalancer) and isinstance(replica.load_balancer, LeastResponseTimeLoadBalancer):
                replica.load_balancer.wrr_providers = copy.deepcopy(closest_replica.load_balancer.wrr_providers)


        replica.simulator = FunctionSimulator()
        replica.pod = self.create_pod(ld, fn)
        if target_node is not None:
            replica.pod.spec.labels['osmotic-scheduling-target'] = target_node.name
        else:
            replica.pod.spec.labels['osmotic-seed'] = 'True'
            # if no node is specified, there is a seed node
        return replica

    def _find_replicas_by_node_name(self, lb_name: str, nodes_names: List[str]) -> List[LoadBalancerReplica]:
        replicas = []
        for replica in self.get_lb_replicas(lb_name, state=FunctionState.RUNNING):
            if replica.node.ether_node.name in nodes_names:
                replicas.append(replica)
        return replicas

    def osmotic_scale_down_lb(self, lb_name: str, target_nodes: List[Node]):
        # todo work out how to scale down on low enough pressures
        current_replica_count = len(self.get_lb_replicas(lb_name, state=FunctionState.RUNNING))
        if len(target_nodes) >= current_replica_count:
            # make sure we don't stop all load balancers by accident. At least one has to keep running
            target_nodes = target_nodes[:current_replica_count - 1]
        replicas_to_remove = self._find_replicas_by_node_name(lb_name, list(map(lambda n: n.name, target_nodes)))
        for r in replicas_to_remove:
            logger.info(f'Removing lb replica from node {r.node.name}')
            self.lb_finder.remove(r)
            yield from self.remove_lb_replica(r)

    def deploy_lb(self, ld: LoadBalancerDeployment):
        if ld.name in self.lb_deployments:
            raise ValueError('LB function already present')
        self.lb_deployments[ld.name] = ld
        # set up scaler
        scaler = self.lb_scaler_factory.create(ld, self.env)
        self.lb_scalers[ld.name] = scaler
        self.env.process(self.lb_scalers[ld.name].run())
        self.env.metrics.log_function_deployment(ld)
        self.env.metrics.log_function_deployment_lifecycle(ld, 'deploy')
        logger.info(f'deploying seed load balancer {ld.name}')
        yield from self.osmotic_scale_up_lb(ld.name, [None])



def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)