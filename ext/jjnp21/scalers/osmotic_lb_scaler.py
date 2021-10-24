import logging
from collections import defaultdict

from ext.jjnp21.core import LoadBalancerDeployment
# from ext.jjnp21.localized_osmotic_lb_system import OsmoticLoadBalancerCapableFaasSystem
from ext.jjnp21.scalers.lb_scaler import LoadBalancerScaler
from sim.core import Environment
from sim.faas import FunctionState

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
        faas = env.faas


        while self.running:
            yield env.timeout(self.alert_window)
            pressures = faas.calculate_pressures()
            active_lb_node_names = []

            # todo just for debugging. remove later
            lb_nodes_by_city = defaultdict(lambda: 0)
            for replica in faas.get_lb_replicas(self.fn.name):
                if replica.state == FunctionState.RUNNING:
                    lb_nodes_by_city[replica.node.ether_node.labels['city']] += 1
            logger.info(f'Load balancer city distribution: {lb_nodes_by_city}')

            replica_nodes_by_city = defaultdict(lambda: defaultdict(lambda: 0))
            for fn_name, replicas in faas.replicas.items():
                for r in replicas:
                    if r.state == FunctionState.RUNNING:
                        replica_nodes_by_city[fn_name][r.node.ether_node.labels['city']] += 1
            for fn_name, results in replica_nodes_by_city.items():
                replica_nodes_by_city[fn_name] = dict(results)
            replica_nodes_by_city = dict(replica_nodes_by_city)
            logger.info(f'Function replica distribution: {str(replica_nodes_by_city)}')

            client_nodes_by_city = defaultdict(lambda: 0)
            for client in faas.client_nodes:
                client_nodes_by_city[client.labels['city']] += 1
            client_nodes_by_city = dict(client_nodes_by_city)
            logger.info(f'Client distribution: {client_nodes_by_city}')
            # end of debugging output

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
            if len(scale_down_list) > 0 and len(active_lb_node_names) > len(scale_down_list):
                logger.info(f'Removing load balancers from the following nodes based on osmotic pressure: {scale_down_list}')
                yield from faas.osmotic_scale_down_lb(self.fn.name, scale_down_list)
            logger.info(f'Ran osmotic scaling check. Added {len(scale_up_list)} replicas, removed {len(scale_down_list)}')
            logger.info(f'Currently {len(active_lb_node_names)} load balancers running')
            # todo: for my info/rememberance: Next check which region the function replicas are located in.
            # my guess is that they are quite concentrated and that this is the reason for the low other pressures.
            # or it is the regions where there are very few function replicas, but very close that score so super highly
            # that could be the case too...

            # scale_up_list

    def stop(self):
        self.running = False
