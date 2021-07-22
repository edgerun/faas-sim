from ext.jjnp21.scalers.lb_scaler import LoadBalancerScaler
from ext.jjnp21.topology import get_non_client_nodes
from sim.core import Environment
from sim.faas import FunctionDeployment, FaasSystem, FunctionState


class FractionScaler(LoadBalancerScaler):
    def __init__(self, fn: FunctionDeployment, env: Environment, target_fraction: float = 0.1):
        self.env = env
        self.function_invocations = dict()
        self.threshold = fn.scaling_config.target_average_rps
        self.alert_window = fn.scaling_config.alert_window
        self.target_fraction = target_fraction
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: FaasSystem = env.faas
        while self.running:
            yield env.timeout(self.alert_window)
            if self.function_invocations.get(self.fn_name, None) is None:
                self.function_invocations[self.fn_name] = 0
            running_replicas = faas.get_replicas(self.fn.name, FunctionState.RUNNING)
            running = len(running_replicas)
            if running == 0:
                continue

            conceived_replicas = faas.get_replicas(self.fn.name, FunctionState.CONCEIVED)
            starting_replicas = faas.get_replicas(self.fn.name, FunctionState.STARTING)

            node_count = len(get_non_client_nodes(self.env.topology))
            desired_replicas = min(round(node_count * self.target_fraction), self.fn.scaling_config.scale_max)

            updated_desired_replicas = desired_replicas
            if len(conceived_replicas) > 0 or len(starting_replicas) > 0:
                if desired_replicas > len(running_replicas):
                    count = len(conceived_replicas) + len(starting_replicas)
                    updated_desired_replicas -= count

            if desired_replicas > len(running_replicas) and updated_desired_replicas < len(running_replicas):
                # no scaling in case of reversed decision
                continue

            if desired_replicas < len(running_replicas):
                # scale down
                scale = len(running_replicas) - desired_replicas
                yield from faas.scale_down(self.fn.name, scale)
            else:
                # scale up
                scale = desired_replicas - len(running_replicas)
                yield from faas.scale_up(self.fn.name, scale)

    def stop(self):
        self.running = False
