from typing import Dict

from ext.raith21 import images
from sim.faas import FunctionCharacterization
from sim.oracle.oracle import ResourceOracle, FetOracle


def get_raith21_function_characterizations(resource_oracle: ResourceOracle,
                                           fet_oracle: FetOracle) -> Dict[str, FunctionCharacterization]:
    return {
        images.resnet50_inference_cpu_manifest: FunctionCharacterization(
            images.resnet50_inference_cpu_manifest, fet_oracle,
            resource_oracle),
        images.resnet50_inference_gpu_manifest: FunctionCharacterization(images.resnet50_inference_gpu_manifest,
                                                                         fet_oracle,
                                                                         resource_oracle),
        images.speech_inference_gpu_manifest: FunctionCharacterization(images.speech_inference_gpu_manifest, fet_oracle,
                                                                       resource_oracle),
        images.speech_inference_tflite_manifest: FunctionCharacterization(images.speech_inference_tflite_manifest,
                                                                          fet_oracle, resource_oracle),
        images.mobilenet_inference_tflite_manifest: FunctionCharacterization(
            images.mobilenet_inference_tflite_manifest,
            fet_oracle,
            resource_oracle),
        images.mobilenet_inference_tpu_manifest: FunctionCharacterization(images.mobilenet_inference_tpu_manifest,
                                                                          fet_oracle, resource_oracle),
        images.resnet50_training_gpu_manifest: FunctionCharacterization(images.resnet50_training_gpu_manifest,
                                                                        fet_oracle,
                                                                        resource_oracle),

        images.resnet50_training_cpu_manifest: FunctionCharacterization(images.resnet50_training_cpu_manifest,
                                                                        fet_oracle,
                                                                        resource_oracle),
        images.tf_gpu_manifest: FunctionCharacterization(images.tf_gpu_manifest,
                                                         fet_oracle,
                                                         resource_oracle),
        images.pi_manifest: FunctionCharacterization(images.pi_manifest,
                                                     fet_oracle,
                                                     resource_oracle),
        images.fio_manifest: FunctionCharacterization(images.fio_manifest,
                                                      fet_oracle,
                                                      resource_oracle),
        images.resnet50_preprocessing_manifest: FunctionCharacterization(images.resnet50_preprocessing_manifest,
                                                                         fet_oracle,
                                                                         resource_oracle)

    }
