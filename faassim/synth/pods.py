from typing import Generator, Dict, Tuple

from skippy.core.model import Pod, PodSpec, Container, ResourceRequirements, ImageState
from skippy.core.utils import parse_size_string

from faassim.stats import BufferedSampler, ScaledParetoSampler, IntegerTruncationSampler
from faassim.synth.images import ImageSynthesizer

PodSynthesizer = Generator[Pod, None, None]


def pod_synthesizer() -> PodSynthesizer:
    cnt = 1
    creators = [create_ml_wf_1_pod, create_ml_wf_2_pod, create_ml_wf_3_serve]
    while True:
        # Rotate over the different workflow functions
        pod = creators[(cnt - 1) % len(creators)](cnt)
        cnt += 1
        yield pod


def create_ml_wf_1_pod(cnt: int) -> Pod:
    return create_pod(cnt,
                      image_name='alexrashed/ml-wf-1-pre:0.37',
                      memory='100Mi',
                      labels={
                          'data.skippy.io/receives-from-storage': '12Mi',
                          'data.skippy.io/sends-to-storage': '209Mi'
                      })


def create_ml_wf_2_pod(cnt: int) -> Pod:
    return create_pod(cnt,
                      image_name='alexrashed/ml-wf-2-train:0.37',
                      memory='1Gi',
                      labels={
                          'capability.skippy.io/nvidia-cuda': '10',
                          'capability.skippy.io/nvidia-gpu': '',
                          'data.skippy.io/receives-from-storage': '209Mi',
                          'data.skippy.io/sends-to-storage': '1500Ki'
                      })


def create_ml_wf_3_serve(cnt: int) -> Pod:
    return create_pod(cnt,
                      image_name='alexrashed/ml-wf-3-serve:0.37',
                      labels={
                          'data.skippy.io/receives-from-storage': '1500Ki'
                      })


def create_pod(cnt: int, image_name: str, memory: str = None, cpu: int = None, labels: Dict[str, str] = None) -> Pod:
    spec = PodSpec()
    resource_requirements = ResourceRequirements()
    if memory:
        resource_requirements.requests['memory'] = parse_size_string(memory)
    if cpu:
        resource_requirements.requests['cpu'] = cpu
    container = Container(image_name, resource_requirements)
    spec.containers = [container]
    spec.labels = labels
    pod = Pod('pod-{0}'.format(cnt), 'openfaas-fn')
    pod.spec = spec
    return pod


class MLWorkflowPodSynthesizer:

    def __init__(self, max_image_variety=None, pareto=True, image_synthesizer=None) -> None:
        super().__init__()
        self.max_image_variety = max_image_variety

        if pareto:
            # 80% of pods will be assigned the same 20% of images
            self.image_id_sampler = BufferedSampler(IntegerTruncationSampler(ScaledParetoSampler(0, max_image_variety)))
        else:
            self.image_id_sampler = None

        self.image_synthesizer = image_synthesizer or ImageSynthesizer()

    def get_image_states(self) -> Dict[str, ImageState]:
        return self.image_synthesizer.get_image_states()

    def create_workflow_pods(self, instance_id) -> Tuple[Pod, Pod, Pod]:
        image_id = self.get_image_id(instance_id)

        return (
            self.create_ml_wf_1_pod(instance_id, image_id, pod_id=instance_id * 3 + 0),
            self.create_ml_wf_2_pod(instance_id, image_id, pod_id=instance_id * 3 + 1),
            self.create_ml_wf_3_pod(instance_id, image_id, pod_id=instance_id * 3 + 2)
        )

    def get_image_id(self, instance_id):
        if self.image_id_sampler:
            image_id = self.image_id_sampler.sample()
        else:
            image_id = instance_id % self.max_image_variety if self.max_image_variety else instance_id
        return image_id

    def create_ml_wf_1_pod(self, instance_id: int, image_id: int, pod_id: int) -> Pod:
        image, _ = self.image_synthesizer.create_ml_wf_1_image(image_id)

        raw_data = f'bucket_{instance_id}/raw_data'
        train_data = f'bucket_{instance_id}/train_data'

        return create_pod(pod_id,
                          image_name=image,
                          memory='100Mi',
                          labels={
                              'data.skippy.io/receives-from-storage': '12Mi',
                              'data.skippy.io/sends-to-storage': '209Mi',
                              'data.skippy.io/receives-from-storage/path': raw_data,
                              'data.skippy.io/sends-to-storage/path': train_data
                          })

    def create_ml_wf_2_pod(self, instance_id: int, image_id: int, pod_id: int) -> Pod:
        image, _ = self.image_synthesizer.create_ml_wf_2_image(image_id)

        train_data = f'bucket_{instance_id}/train_data'
        serialized_model = f'bucket_{instance_id}/model'

        return create_pod(pod_id,
                          image_name=image,
                          memory='1Gi',
                          labels={
                              'capability.skippy.io/nvidia-cuda': '10',
                              'capability.skippy.io/nvidia-gpu': '',
                              'data.skippy.io/receives-from-storage': '209Mi',
                              'data.skippy.io/sends-to-storage': '1500Ki',
                              'data.skippy.io/receives-from-storage/path': train_data,
                              'data.skippy.io/sends-to-storage/path': serialized_model
                          })

    def create_ml_wf_3_pod(self, instance_id: int, image_id: int, pod_id: int) -> Pod:
        image, _ = self.image_synthesizer.create_ml_wf_3_image(image_id)
        serialized_model = f'bucket_{instance_id}/model'

        return create_pod(pod_id,
                          image_name=image,
                          labels={
                              'data.skippy.io/receives-from-storage': '1500Ki',
                              'data.skippy.io/receives-from-storage/path': serialized_model
                          })
