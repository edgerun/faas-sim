import logging

from ext.jjnp21.core import LoadBalancerDeployment
from ext.jjnp21.localized_osmotic_lb_system import OsmoticLoadBalancerCapableFaasSystem
from ext.jjnp21.scalers.lb_scaler import LoadBalancerScaler
from sim.core import Environment

logger = logging.getLogger(__name__)


class OsmoticLoadBalancerScaler(LoadBalancerScaler):
    def __init__(self, fn: LoadBalancerDeployment, env: Environment, pressure_threshold: float, hysteresis: float):
        self.fn = fn
        self.env = env
        self.pressure_threshold = pressure_threshold
        self.hysteresis = hysteresis
        self.alert_window = fn.scaling_config.alert_window
        self.running = True
        pass

    def run(self):
        logger.info('Running osmotic scaling round')
        env = self.env
        faas: OsmoticLoadBalancerCapableFaasSystem = None
        if isinstance(env.faas, OsmoticLoadBalancerCapableFaasSystem):
            faas = env.faas
        else:
            raise ValueError(
                'FaaS system is not osmotic load balancer capable, although osmotic load balancer scaler is used')

        while self.running:
            yield env.timeout
            pressures = faas.calculate_pressures()
            active_lb_node_names = []
            for replica in faas.get_lb_replicas(self.fn.name):
                active_lb_node_names.append(replica.node.ether_node.name)
            scale_up_list = []
            scale_down_list = []
            for node, pressure in pressures.items():
                if pressure >= self.pressure_threshold + self.hysteresis and node.name not in active_lb_node_names:
                    scale_up_list.append(node)
                elif pressure <= self.pressure_threshold - self.hysteresis and node.name in active_lb_node_names:
                    scale_down_list.append(node)

            # If no node is running yet and there is no pressure suggesting that it needs to run: start a seed load balancer
            if len(active_lb_node_names) == 0 and len(scale_up_list) == 0:
                scale_up_list.append(None)

            if len(scale_up_list) > 0:
                logger.info(f'Adding load balancers on the following nodes based on osmotic pressure: {scale_up_list}')
                yield from faas.osmotic_scale_up_lb(self.fn.name, scale_up_list)
            if len(scale_down_list) > 0:
                logger.info(f'Removing load balancers from the following nodes based on osmotic pressure: {scale_down_list}')
                yield from faas.osmotic_scale_down_lb(self.fn.name, scale_down_list)
            logger.info(f'Ran osmotic scaling check. Added {len(scale_up_list)} replicas, removed {len(scale_down_list)}')
            # scale_up_list

    def stop(self):
        self.running = False
