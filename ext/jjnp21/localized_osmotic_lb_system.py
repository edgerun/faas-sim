from ext.jjnp21.automator.factories.lb_scaler import LoadBalancerScalerFactory
from ext.jjnp21.localized_lb_system import LocalizedLoadBalancerFaasSystem
from sim.core import Environment
from sim.faas import FunctionRequest


class OsmoticLoadBalancerCapableFaasSystem(LocalizedLoadBalancerFaasSystem):
    def __init__(self, env: Environment, lb_scaler_factory: LoadBalancerScalerFactory,
                 scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False,
                 scale_static: bool = False, lb_osmotic_pressure_threshold: float = 1.0,
                 lb_osmotic_pressure_hysteresis: float = 0.1):
        super().__init__(env, lb_scaler_factory, scale_by_requests, scale_by_average_requests,
                         scale_by_queue_requests_per_replica, scale_static)
        self.lb_osmotic_pressure_threshold = lb_osmotic_pressure_hysteresis
        self.lb_osmotic_pressure_hysteresis = lb_osmotic_pressure_hysteresis

    def calculate_pressure(self, node):
        pass
        """
        TODO: get clients that would be attached to node if it were a LB
         - for each client get the LB
         - if distance client-LB > than client-node, then node gets the client
         - get the request share the client would have per function
         - get the list of replicas for each function
         - calculate the distance from the replicas and weigh it appropriately
        """

    def set_request_client(self, request: FunctionRequest):
        # todo record the metrics we require for the scaling stuff
        super().set_request_client(request)

    def osmotic_scale_up_lb(self, lb_name: str, target_node: str):
        # todo: create pod and everything with labels according to the params
        pass

    def osmotic_scale_down_lb(self, lb_name: str, target_node: str):
        # todo work out how to scale down on low enough pressures
        pass

    def scale_up_lb(self, lb_name: str, add_count: int):
        return super().scale_up_lb(lb_name, add_count)

    def scale_down_lb(self, lb_name: str, remove_count: int):
        super().scale_down_lb(lb_name, remove_count)
