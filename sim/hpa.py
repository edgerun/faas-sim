import math

from sim.core import Environment
from sim.faas import FunctionState, FaasSystem
from sim.resource import MetricsServer



class HorizontalPodAutoscaler:

    # Behavior and default values taken from:
    # https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    # Official implementation only considers pods, in which all containers have specified resource requests

    def __init__(self, env: Environment, average_window: int = 100, reconcile_interval: int = 15,
                 target_tolerance: float = 0.1):
        """
        :param average_window: seconds to look back in time to calculate the average for each replica
        :param env: sim environment
        :param reconcile_interval: wait time for control loop
        :param target_tolerance: determines how close the target/current resource ratio must be to 1.0 to skip scaling
        """
        self.average_window = 100
        self.env = env
        self.reconcile_interval = reconcile_interval
        self.target_tolerance = target_tolerance

    def run(self):
        """


        For each Function Deployment sum up the CPU usage of each running replica and take the mean.

        While the official implementation just uses the CPU usage reported by the metrics server,
        there is no option/default for the corresponding utilization window.
        Therefore, this implementation allows users to set the window size that will be used when querying the
        MetricsServer

        This implementation considers the target value to be a 'targetAverageUtilization', because our MetricsServer
        only can calculate the average utilization of on replica and has no means to report exact values (i.e.: millis)
        Also, the official HPA calculates average relative to the requested resource.
        We do this relative to the maximum (1.0 == 100% Utilization of all cores)

        Further, copied from docs:
        "If there were any missing metrics, we recompute the average more conservatively, assuming those pods
        were consuming 100% of the desired value in case of a scale down, and 0% in case of a scale up.
        This dampens the magnitude of any potential scale <-(!) not implemented, as "missing metrics" can't exist in sim

        Furthermore, if any not-yet-ready pods were present, and we would have scaled up without factoring
        in missing metrics  or not-yet-ready pods, we conservatively assume the not-yet-ready pods are consuming 0%
        of the desired metric, further dampening the magnitude of a scale up. <- implemented, considering: conceived
        and starting nodes to be 'not-yet-ready'.

        After factoring in the not-yet-ready pods and missing metrics, we recalculate the usage ratio. If the new ratio
        reverses the scale direction, or is within the tolerance, we skip scaling. Otherwise, we use the new ratio
        to scale. <- implemented"

        Raw Calculation:
        desiredReplicas = ceil[currentReplicas * ( currentMetricValue / desiredMetricValue )]

        """
        while True:
            yield self.env.timeout(self.reconcile_interval)
            metrics_server: MetricsServer = self.env.metrics_server
            faas: FaasSystem = self.env.faas
            for function_deployment in faas.functions_deployments.values():
                running_replicas = faas.get_replicas(function_deployment.name, FunctionState.RUNNING)
                if len(running_replicas) == 0:
                    continue
                conceived_replicas = faas.get_replicas(function_deployment.name, FunctionState.CONCEIVED)
                starting_replicas = faas.get_replicas(function_deployment.name, FunctionState.STARTING)
                sum_cpu = 0

                for replica in running_replicas:
                    sum_cpu += metrics_server.get_average_cpu_utilization(replica, self.average_window)

                average_cpu = sum_cpu / len(running_replicas)

                desired_replicas = math.ceil(
                    len(running_replicas) * (average_cpu / function_deployment.target_average_utilization))

                updated_desired_replicas = desired_replicas
                if len(conceived_replicas) > 0 or len(starting_replicas) > 0:
                    if desired_replicas > len(running_replicas):
                        count = len(running_replicas) + len(conceived_replicas) + len(starting_replicas)
                        average_cpu = sum_cpu / count
                        updated_desired_replicas = math.ceil(
                            len(running_replicas) * (average_cpu / function_deployment.target_average_utilization))

                if desired_replicas > len(running_replicas) and updated_desired_replicas < len(running_replicas):
                    # no scaling in case of reversed decision
                    continue

                ratio = average_cpu / function_deployment.target_average_utilization
                if 1 > ratio >= 1 - self.target_tolerance:
                    # ratio is sufficiently close to 1.0
                    continue

                if 1 < ratio < 1 + self.target_tolerance:
                    continue

                if desired_replicas < len(running_replicas):
                    # scale down
                    scale = len(running_replicas) - desired_replicas
                    yield from faas.scale_down(function_deployment.name, scale)
                else:
                    # scale up
                    scale = desired_replicas - len(running_replicas)
                    yield from faas.scale_up(function_deployment.name, scale)
