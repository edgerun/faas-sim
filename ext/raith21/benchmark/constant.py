from ext.raith21 import images
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.oracles import Raith21FetOracle, Raith21ResourceOracle
from ext.raith21.resources import ai_resources_per_node_image
from ext.raith21.utils import create_deployments_for_profile
from sim.benchmark import BenchmarkBase, set_degradation
from sim.core import Environment
from sim.requestgen import expovariate_arrival_profile, constant_rps_profile


class ConstantBenchmark(BenchmarkBase):

    def __init__(self, profile: str, duration: int, rps=200, model_folder=None):
        all_images = images.all_ai_images
        self.model_folder = model_folder
        self.profile = profile
        self.rps = rps
        self.duration = duration
        fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
        resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)

        deployments = create_deployments_for_profile(profile, fet_oracle, resource_oracle)

        super().__init__(all_images, list(deployments.values()), arrival_profiles=dict(), duration=duration)

    @property
    def settings(self):
        return {
            'profile': self.profile,
            'rps': self.rps,
            'duration': self.duration
        }

    @property
    def type(self):
        return 'constant'

    def setup(self, env: Environment):
        self.set_deployments(env)
        self.setup_profile()
        if self.model_folder is not None:
            set_degradation(env, self.model_folder)
        super().setup(env)

    def setup_profile(self):
        if self.profile == 'service':
            self.set_service_profiles()
        elif self.profile == 'ai':
            self.set_ai_profiles()
        elif self.profile == 'mixed':
            self.set_mixed_profiles()
        else:
            raise AttributeError(f'unknown profile: {self.profile}')

    def set_mixed_profiles(self):
        self.arrival_profiles[images.resnet50_inference_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))

        self.arrival_profiles[images.mobilenet_inference_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))

        self.arrival_profiles[images.speech_inference_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))

        self.arrival_profiles[images.resnet50_training_function] = \
            expovariate_arrival_profile(constant_rps_profile(0.1))

        self.arrival_profiles[images.resnet50_preprocessing_function] = \
            expovariate_arrival_profile(constant_rps_profile(1))

    def set_ai_profiles(self):

        self.arrival_profiles[images.resnet50_inference_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))

        self.arrival_profiles[images.mobilenet_inference_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))

        self.arrival_profiles[images.speech_inference_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))

        self.arrival_profiles[images.resnet50_training_function] = \
            expovariate_arrival_profile(constant_rps_profile(0.1))

        self.arrival_profiles[images.resnet50_preprocessing_function] = \
            expovariate_arrival_profile(constant_rps_profile(1))

    def set_service_profiles(self):
        self.arrival_profiles[images.pi_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))
        self.arrival_profiles[images.tf_gpu_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))
        # set arrival profiles
        self.arrival_profiles[images.fio_function] = \
            expovariate_arrival_profile(constant_rps_profile(self.rps))

    def set_deployments(self, env):
        deployments = self.deployments_per_name
        for deployment in deployments.values():
            deployment.scale_min = 5
            deployment.target_average_utilization = 0.5
        no_of_devices = len(env.topology.get_nodes())

        deployments[images.resnet50_inference_function].rps_threshold = 100
        deployments[images.resnet50_inference_function].scale_max = int(0.7 * no_of_devices)
        deployments[images.resnet50_inference_function].scale_factor = int(0.05 * no_of_devices)
        deployments[images.resnet50_inference_function].rps_threshold_duration = 10

        deployments[images.mobilenet_inference_function].rps_threshold = 70
        deployments[images.mobilenet_inference_function].scale_max = int(0.25 * no_of_devices)
        deployments[images.mobilenet_inference_function].scale_factor = 5
        deployments[images.mobilenet_inference_function].rps_threshold_duration = 10

        deployments[images.speech_inference_function].rps_threshold = 40
        deployments[images.speech_inference_function].scale_max = int(0.25 * no_of_devices)
        deployments[images.speech_inference_function].scale_factor = 5
        deployments[images.speech_inference_function].rps_threshold_duration = 15

        deployments[images.resnet50_preprocessing_function].rps_threshold = 40
        deployments[images.resnet50_preprocessing_function].scale_max = no_of_devices / 4
        deployments[images.resnet50_preprocessing_function].scale_factor = 1
        deployments[images.resnet50_preprocessing_function].rps_threshold_duration = 15

        deployments[images.resnet50_training_function].rps_threshold = 40
        deployments[images.resnet50_training_function].scale_max = no_of_devices / 2
        deployments[images.resnet50_training_function].scale_factor = 1
        deployments[images.resnet50_training_function].rps_threshold_duration = 15
