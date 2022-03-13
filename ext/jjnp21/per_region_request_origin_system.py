from collections import defaultdict
import random

from ext.jjnp21.automator.factories.lb_scaler import LoadBalancerScalerFactory
from ext.jjnp21.localized_lb_system import NetworkSimulationMode
from ext.jjnp21.localized_osmotic_lb_system import OsmoticLoadBalancerCapableFaasSystem
from sim.core import Environment, Node
from sim.faas import FunctionRequest
from typing import Dict, Any, List, Tuple


# Extension that allows us to model request origins changing origin locations (cities)
class PerRegionRequestOriginFaasSystem(OsmoticLoadBalancerCapableFaasSystem):

    def __init__(self, env: Environment, lb_scaler_factory: LoadBalancerScalerFactory,
                 scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False,
                 scale_static: bool = False,
                 net_mode: NetworkSimulationMode = NetworkSimulationMode.ACCURATE,
                 lb_osmotic_pressure_window_size: float = 60,
                 region_weight_mapping: Dict[str, Any] = {}):
        super().__init__(env, lb_scaler_factory, scale_by_requests, scale_by_average_requests,
                         scale_by_queue_requests_per_replica, scale_static, net_mode, lb_osmotic_pressure_window_size)

        # region weight mapping maps the region (i.e. city) string key to a weight generator function
        # that weight generator function takes the current time as input and returns the relative chance of the region emitting the next request
        # the region is chosen on a weighted random based on the weights resulting from that
        self.region_weight_mapping = region_weight_mapping
        self.region_bins: Dict[str, List[Node]] = defaultdict(list)
        for client in self.client_nodes:
            self.region_bins[client.labels['city']].append(client)
        self.cities_list = list(region_weight_mapping.keys())

        # here we should also generate the buckets separating clients by their region/city -> saves computations later

    # todos:
    # - build buckets of cities that are available
    # - build

    def get_current_city_weight_set(self) -> List[float]:
        return list(map(lambda x: x[1](self.env.now), self.region_weight_mapping.items()))

    def sample_client(self) -> Node:
        chosen_city = random.choices(population=self.cities_list, weights=self.get_current_city_weight_set(), k=1)[0]
        return random.choice(self.region_bins[chosen_city])

    # Overrides the method, using this new type
    def set_request_client(self, request: FunctionRequest):

        if len(self.client_nodes) > 0:
            client = self.sample_client()
            request.client_node = client
        self.request_log[request.name][request.client_node].append(self.env.now)
