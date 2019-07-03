from typing import Generator

from core.model import Pod, PodSpec, Container, ResourceRequirements

PodSynthesizer = Generator[Pod, None, None]


def pod_synthesizer() -> PodSynthesizer:
    # TODO: synthesize parameters, add function specific annotations based on our scenario
    cnt = 1
    while True:
        spec = PodSpec()
        container = Container('alexrashed/ml-wf-1-pre:0.33', ResourceRequirements())
        spec.containers = [container]

        pod = Pod('pod-{0}'.format(cnt), 'openfaas-fn')
        cnt += 1
        pod.spec = spec

        yield pod
