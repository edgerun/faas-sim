from typing import Dict, List

from ext.jjnp21.core import LoadBalancerDeployment
from ext.jjnp21.load_balancers.localized import LocalizedLRT, LocalizedRR
from ext.jjnp21.load_balancers.lrt import LeastResponseTimeLoadBalancer
from ext.jjnp21.misc import images
from sim.core import Environment
from sim.faas import LoadBalancer, FunctionReplica, FunctionImage, Function, KubernetesResourceConfiguration, \
    FunctionContainer, ScalingConfiguration, RoundRobinLoadBalancer

default_scaling_config = ScalingConfiguration()
default_scaling_config.scale_max = 1000


class LRTLoadBalancerDeployment(LoadBalancerDeployment):
    def __init__(self, lrt_window: float = 15, weight_update_frequency: float = 15,
                 scaling_config: ScalingConfiguration = default_scaling_config):
        self.lrt_window = lrt_window
        self.weight_update_frequency = weight_update_frequency
        # we have only one "image" since it only runs on CPU and the cpu architectures don't count as different images
        function_image = FunctionImage(image=images.traefik_lrt_manifest)
        function = Function(images.traefik_lrt_function, fn_images=[function_image])
        # todo: enter resource values from experiments for LB instead
        kube_resource_config = KubernetesResourceConfiguration.create_from_str(cpu="1000m", memory="500Mi")
        function_container = FunctionContainer(
            function_image,
            resource_config=kube_resource_config
        )
        super().__init__(function, [function_container], scaling_config)

    def create_load_balancer(self, env: Environment, replicas: Dict[str, List[FunctionReplica]]) -> LoadBalancer:
        return LocalizedLRT(env, replicas, lrt_window=self.lrt_window,
                            weight_update_frequency=self.weight_update_frequency)


class RRLoadBalancerDeployment(LoadBalancerDeployment):
    def __init__(self, scaling_config: ScalingConfiguration = default_scaling_config):
        # we have only one "image" since it only runs on CPU and the cpu architectures don't count as different images
        function_image = FunctionImage(image=images.traefik_rr_manifest)
        function = Function(images.traefik_rr_function, fn_images=[function_image])
        # todo: enter resource values from experiments for LB instead
        kube_resource_config = KubernetesResourceConfiguration.create_from_str(cpu="1000m", memory="500Mi")
        function_container = FunctionContainer(
            function_image,
            resource_config=kube_resource_config
        )
        super().__init__(function, [function_container], scaling_config)

    def create_load_balancer(self, env: Environment, replicas: Dict[str, List[FunctionReplica]]) -> LoadBalancer:
        return LocalizedRR(env, replicas)
