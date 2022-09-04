from typing import List

from faas.system import FunctionDeployment, FunctionContainer, Function, ScalingConfiguration, DeploymentRanking


class SimScalingConfiguration:
    scaling_config: ScalingConfiguration

    def __init__(self, scaling_config: ScalingConfiguration = None):
        self.scaling_config = scaling_config
        if scaling_config is None:
            self.scaling_config = ScalingConfiguration()

    @property
    def scale_min(self):
        return self.scaling_config.scale_min

    @property
    def scale_max(self):
        return self.scaling_config.scale_max

    @property
    def scale_zero(self):
        return self.scaling_config.scale_zero

    @property
    def scale_factor(self):
        return self.scaling_config.scale_factor

    # average requests per second threshold for scaling
    rps_threshold: int = 20

    # window over which to track the average rps
    alert_window: int = 50  # TODO currently not supported by FaasRequestScaler

    # seconds the rps threshold must be violated to trigger scale up
    rps_threshold_duration: int = 10

    # target average cpu utilization of all replicas, used by HPA
    target_average_utilization: float = 0.5

    # target average rps over all replicas, used by AverageFaasRequestScaler
    target_average_rps: int = 200

    # target of maximum requests in queue
    target_queue_length: int = 75

    target_average_rps_threshold = 0.1

    def __str__(self):
        return str(self.scaling_config)



class SimFunctionDeployment(FunctionDeployment):
    scaling_config: SimScalingConfiguration
    # used to determine which function to take when scaling
    ranking: DeploymentRanking

    def __init__(self, fn: Function, fn_containers: List[FunctionContainer], scaling_config: SimScalingConfiguration,
                 deployment_ranking: DeploymentRanking = None):
        super().__init__(fn, fn_containers, scaling_config, deployment_ranking)
        self.scaling_config = scaling_config
        if deployment_ranking is None:
            self.ranking = DeploymentRanking(fn_containers)
        else:
            self.ranking = deployment_ranking

    def get_selected_service(self):
        return self.fn.get_image(self.ranking.get_first())
