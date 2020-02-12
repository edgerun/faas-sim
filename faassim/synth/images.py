from typing import Dict, Tuple

from core.model import ImageState

# TODO: introduce distribution samplers for image size
templates = {
    'alexrashed/ml-wf-1-pre:{image_id}': ImageState(size={
        'arm': 465830200,
        'arm64': 540391110,
        'amd64': 533323136
    }),
    'alexrashed/ml-wf-2-train:{image_id}': ImageState(size={
        'arm': 519336111,
        'arm64': 594174340,
        'amd64': 550683347
    }),
    'alexrashed/ml-wf-3-serve:{image_id}': ImageState(size={
        'arm': 511888808,
        'arm64': 590989596,
        'amd64': 589680790
    })
}


class ImageSynthesizer:
    image_states: Dict[str, ImageState]

    def __init__(self) -> None:
        super().__init__()
        self.image_states = dict()

    def get_image_states(self) -> Dict[str, ImageState]:
        return self.image_states

    def create_image(self, template_name: str, image_id: int) -> Tuple[str, ImageState]:
        name = template_name.format(image_id=image_id)

        if name not in self.image_states:
            image_state = templates[template_name]
            self.image_states[name] = image_state

        return name, self.image_states[name]

    def create_ml_wf_1_image(self, image_id: int) -> Tuple[str, ImageState]:
        return self.create_image('alexrashed/ml-wf-1-pre:{image_id}', image_id)

    def create_ml_wf_2_image(self, image_id: int) -> Tuple[str, ImageState]:
        return self.create_image('alexrashed/ml-wf-2-train:{image_id}', image_id)

    def create_ml_wf_3_image(self, image_id: int) -> Tuple[str, ImageState]:
        return self.create_image('alexrashed/ml-wf-3-serve:{image_id}', image_id)
