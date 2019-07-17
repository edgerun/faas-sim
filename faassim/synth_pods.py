from typing import Generator, Dict

from core.model import Pod, PodSpec, Container, ResourceRequirements
from core.utils import parse_size_string

PodSynthesizer = Generator[Pod, None, None]


def pod_synthesizer() -> PodSynthesizer:
    cnt = 1
    creators = [create_ml_wf_1_pod, create_ml_wf_2_pod, create_ml_wf_3_serve]
    while True:
        # Rotate over the different workflow functions
        pod = creators[(cnt - 1) % 3](cnt)
        cnt += 1
        yield pod


def create_ml_wf_1_pod(cnt: int) -> Pod:
    return create_pod(cnt,
                      image_name='alexrashed/ml-wf-1-pre:0.33',
                      memory='100Mi',
                      labels = {
                            'data.skippy.io/receives-from-storage': '12Mi',
                            'data.skippy.io/sends-to-storage': '209Mi'
                        })


def create_ml_wf_2_pod(cnt: int) -> Pod:
    return create_pod(cnt,
                      image_name='alexrashed/ml-wf-2-train:0.33',
                      memory='1Gi',
                      labels={
                          'capability.skippy.io/nvidia-cuda': '10',
                          'capability.skippy.io/nvidia-gpu': '',
                          'data.skippy.io/receives-from-storage': '209Mi',
                          'data.skippy.io/sends-to-storage': '1500Ki'
                        })


def create_ml_wf_3_serve(cnt: int) -> Pod:
    return create_pod(cnt,
                      image_name='alexrashed/ml-wf-3-serve:0.33',
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
