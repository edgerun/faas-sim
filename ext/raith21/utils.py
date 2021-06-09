from typing import Dict

from ext.raith21 import images
from ext.raith21.deployments import create_all_deployments
from sim.faas import FunctionDeployment
from sim.oracle.oracle import FetOracle, ResourceOracle


def extract_model_type(device: str):
    if not type(device) is str:
        return ''
    try:
        return device[:device.rindex('_')]
    except ValueError:
        return device


def create_ai_deployments(fet_oracle: FetOracle, resource_oracle: ResourceOracle) -> Dict[str, FunctionDeployment]:
    all_deployments = create_all_deployments(fet_oracle, resource_oracle)
    del all_deployments[images.tf_gpu_function]
    del all_deployments[images.pi_function]
    del all_deployments[images.fio_function]
    return all_deployments


def create_mixed_deployments(fet_oracle: FetOracle, resource_oracle: ResourceOracle) -> Dict[str, FunctionDeployment]:
    return create_all_deployments(fet_oracle, resource_oracle)


def create_service_deployments(fet_oracle: FetOracle, resource_oracle: ResourceOracle) -> Dict[str, FunctionDeployment]:
    all_deployments = create_all_deployments(fet_oracle, resource_oracle)
    del all_deployments[images.speech_inference_function]
    del all_deployments[images.mobilenet_inference_function]
    del all_deployments[images.resnet50_inference_function]
    del all_deployments[images.resnet50_preprocessing_function]
    del all_deployments[images.resnet50_training_function]
    return all_deployments

def create_inference_deployments(fet_oracle: FetOracle, resource_oracle: ResourceOracle) -> Dict[str, FunctionDeployment]:
    all_deployments = create_all_deployments(fet_oracle, resource_oracle)
    # del all_deployments[images.tf_gpu_function]
    # del all_deployments[images.pi_function]
    # del all_deployments[images.fio_function]
    del all_deployments[images.resnet50_training_function]
    del all_deployments[images.resnet50_preprocessing_function]
    return all_deployments

def create_deployments_for_profile(profile: str, fet_oracle: FetOracle, resource_oracle: ResourceOracle) -> Dict[
    str, FunctionDeployment]:
    if profile == 'ai':
        return create_ai_deployments(fet_oracle, resource_oracle)
    elif profile == 'mixed':
        return create_mixed_deployments(fet_oracle, resource_oracle)
    elif profile == 'service':
        return create_service_deployments(fet_oracle, resource_oracle)
    elif profile == 'inference':
        return create_inference_deployments(fet_oracle, resource_oracle)
    else:
        raise ValueError(f'unknown profile: {profile}')
