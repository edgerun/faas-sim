from typing import Generator

from core.model import Pod, PodSpec

PodSynthesizer = Generator[Pod, None, None]


def pod_synthesizer() -> PodSynthesizer:
    # TODO: synthesize parameters, add function specific annotations,...
    cnt = 1
    while True:
        spec = PodSpec()

        pod = Pod('pod-{0}'.format(cnt))
        cnt += 1
        pod.spec = spec

        yield pod
