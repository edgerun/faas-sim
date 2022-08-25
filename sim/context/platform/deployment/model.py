from typing import List, Dict

from faas.system import FunctionDeployment, FunctionContainer, Function, ScalingConfiguration


class SimScalingConfiguration:
    scaling_config: ScalingConfiguration

    def __init__(self, scaling_config: ScalingConfiguration = None):
        self.scaling_config = ScalingConfiguration()
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
        return self.scale_factor

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


class DeploymentRanking:
    # TODO probably better to remove default/enable default for one image
    images: List[str]

    # TODO probably removable after moving decision on which node to deploy pod to user
    # percentages of scaling per image, can be used to hinder scheduler to overuse expensive resources (i.e. tpu)
    function_factor: Dict[str, float]

    def __init__(self, images: List[str], function_factor: Dict[str, float] = None):
        self.images = images
        self.function_factor = function_factor if function_factor is not None else {image: 1 for image in images}

    def set_first(self, image: str):
        index = self.images.index(image)
        updated = self.images[:index] + self.images[index + 1:]
        self.images = [image] + updated

    def get_first(self):
        return self.images[0]


class SimFunctionDeployment(FunctionDeployment):
    scaling_config: SimScalingConfiguration
    # used to determine which function to take when scaling
    ranking: DeploymentRanking

    def __init__(self, fn: Function, fn_containers: List[FunctionContainer], scaling_config: SimScalingConfiguration,
                 deployment_ranking: DeploymentRanking = None):
        super().__init__(fn, fn_containers, scaling_config, deployment_ranking)
        self.scaling_config = scaling_config
        if deployment_ranking is None:
            self.ranking = DeploymentRanking([x.image for x in self.fn.fn_images])
        else:
            self.ranking = deployment_ranking

    def get_selected_service(self):
        return self.fn.get_image(self.ranking.get_first())
