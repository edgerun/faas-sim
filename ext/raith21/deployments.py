from dataclasses import dataclass
from typing import Dict

from ext.raith21 import images, storage
from sim.faas import Resources, DeploymentRanking, FunctionDeployment, FunctionDefinition, FunctionCharacterization
from sim.oracle.oracle import ResourceOracle, FetOracle

default_resnet_inference_ranking = DeploymentRanking(
    [images.resnet50_inference_gpu_manifest, images.resnet50_inference_cpu_manifest])
default_speech_inference_ranking = DeploymentRanking(
    [images.speech_inference_tflite_manifest, images.speech_inference_gpu_manifest])
default_mobilenet_inference_ranking = DeploymentRanking(
    [images.mobilenet_inference_tpu_manifest, images.mobilenet_inference_tflite_manifest])
default_resnet_training_ranking = DeploymentRanking(
    [images.resnet50_training_gpu_manifest, images.resnet50_training_cpu_manifest])
default_pi_ranking = DeploymentRanking([images.pi_manifest])
default_fio_ranking = DeploymentRanking([images.fio_manifest])
default_tf_gpu_ranking = DeploymentRanking([images.tf_gpu_manifest])
default_resnet_preprocessing_ranking = DeploymentRanking([images.resnet50_preprocessing_manifest])


@dataclass
class DeploymentSettings:
    resnet_inference_ranking: DeploymentRanking = default_resnet_inference_ranking
    resnet_preprocessing_ranking: DeploymentRanking = default_resnet_preprocessing_ranking
    speech_inference_ranking: DeploymentRanking = default_speech_inference_ranking
    mobilenet_inference_ranking: DeploymentRanking = default_mobilenet_inference_ranking
    resnet_training_ranking: DeploymentRanking = default_resnet_training_ranking
    tf_gpu_ranking: DeploymentRanking = default_tf_gpu_ranking
    pi_ranking: DeploymentRanking = default_pi_ranking
    fio_ranking: DeploymentRanking = default_fio_ranking


def get_resnet50_inference_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                                      ranking: DeploymentRanking) -> FunctionDeployment:
    resnet_cpu_characterization = FunctionCharacterization(images.resnet50_inference_cpu_manifest, fet_oracle,
                                                           resource_oracle)
    resnet_gpu_characterization = FunctionCharacterization(images.resnet50_inference_gpu_manifest, fet_oracle,
                                                           resource_oracle)

    model = storage.resnet_model_bucket_item.name
    data_storage = {
        'data.skippy.io/receives-from-storage': '103M',
        'data.skippy.io/receives-from-storage/path': f'{storage.resnet_model_bucket}/{model}',
    }

    resnet50_cpu_function = FunctionDefinition(
        name=images.resnet50_inference_function,
        image=images.resnet50_inference_cpu_manifest,
        characterization=resnet_cpu_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '4a'})

    resnet50_cpu_function.requests = Resources.from_str("150Mi", "1000m")

    resnet50_gpu_function = FunctionDefinition(
        name=images.resnet50_inference_function,
        image=images.resnet50_inference_gpu_manifest,
        characterization=resnet_gpu_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'device.edgerun.io/accelerator': 'GPU',
                'device.edgerun.io/vram': '1500',
                'cluster': '4b'})
    resnet50_gpu_function.requests = Resources.from_str("400Mi", "1000m")
    resnet50_gpu_function.labels.update(data_storage)
    resnet50_cpu_function.labels.update(data_storage)

    deployment = FunctionDeployment(
        images.resnet50_inference_function,
        {
            images.resnet50_inference_gpu_manifest: resnet50_gpu_function,
            images.resnet50_inference_cpu_manifest: resnet50_cpu_function,
        },
        ranking
    )

    deployment.function_factor = {
        images.resnet50_inference_gpu_manifest: 1,
        images.resnet50_inference_cpu_manifest: 1,
    }

    return deployment


def get_speech_inference_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                                    ranking: DeploymentRanking) -> FunctionDeployment:
    speech_inference_gpu_characterization = FunctionCharacterization(images.speech_inference_gpu_manifest, fet_oracle,
                                                                     resource_oracle)
    speech_inference_tflite_characterization = FunctionCharacterization(images.speech_inference_tflite_manifest,
                                                                        fet_oracle, resource_oracle)

    tflite = storage.speech_model_tflite_bucket_item
    data_storage_tflite = {
        'data.skippy.io/receives-from-storage': '48M',
        'data.skippy.io/receives-from-storage/path': f'{storage.speech_bucket}/{tflite.name}',
    }
    gpu = storage.speech_model_gpu_bucket_item
    data_storage_gpu = {
        # this size is without scorer object, which is used to impove accuracy but doesn't seem to affect runtime,
        # scorer weighs around 900M - simple benchmarks in bash have made no difference in runtime
        'data.skippy.io/receives-from-storage': '188M',
        'data.skippy.io/receives-from-storage/path': f'{storage.speech_bucket}/{gpu.name}',

    }

    speech_gpu_function = FunctionDefinition(
        name=images.speech_inference_function,
        image=images.speech_inference_gpu_manifest,
        characterization=speech_inference_gpu_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '0', 'device.edgerun.io/accelerator': 'GPU',
                'device.edgerun.io/vram': '1500', })

    speech_gpu_function.labels.update(data_storage_gpu)
    speech_gpu_function.requests = Resources.from_str("300Mi", "1000m")
    speech_tflite_function = FunctionDefinition(
        name=images.speech_inference_function,
        image=images.speech_inference_tflite_manifest,
        characterization=speech_inference_tflite_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '1'})

    speech_tflite_function.labels.update(data_storage_tflite)
    speech_tflite_function.requests = Resources.from_str("100Mi", "1000m")
    deployment = FunctionDeployment(
        images.speech_inference_function,
        {
            images.speech_inference_gpu_manifest: speech_gpu_function,
            images.speech_inference_tflite_manifest: speech_tflite_function,
        },
        ranking
    )

    deployment.function_factor = {
        images.speech_inference_tflite_manifest: 1,
        images.speech_inference_gpu_manifest: 1
    }

    return deployment


def get_mobilenet_inference_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                                       ranking: DeploymentRanking) -> FunctionDeployment:
    mobilenet_inference_tflite_characterization = FunctionCharacterization(images.mobilenet_inference_tflite_manifest,
                                                                           fet_oracle,
                                                                           resource_oracle)
    mobilenet_inference_tpu_characterization = FunctionCharacterization(images.mobilenet_inference_tpu_manifest,
                                                                        fet_oracle, resource_oracle)

    tflite = storage.mobilenet_model_tflite_bucket_item.name
    data_storage_tflite_labels = {
        'data.skippy.io/receives-from-storage': '4M',
        'data.skippy.io/receives-from-storage/path': f'{storage.mobilenet_bucket}/{tflite}',
    }

    tpu = storage.mobilenet_model_tpu_bucket_item.name
    data_storage_tpu_labels = {
        'data.skippy.io/receives-from-storage': '4M',
        'data.skippy.io/receives-from-storage/path': f'{storage.mobilenet_bucket}/{tpu}',
    }

    mobilenet_tpu_function = FunctionDefinition(
        name=images.mobilenet_inference_function,
        image=images.mobilenet_inference_tpu_manifest,
        characterization=mobilenet_inference_tpu_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '1b', 'device.edgerun.io/accelerator': 'TPU'})
    mobilenet_tpu_function.requests = Resources.from_str("100Mi", "1000m")

    mobilenet_tflite_function = FunctionDefinition(
        name=images.mobilenet_inference_function,
        image=images.mobilenet_inference_tflite_manifest,
        characterization=mobilenet_inference_tflite_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '1a'})
    mobilenet_tflite_function.requests = Resources.from_str("100Mi", "1000m")
    mobilenet_tpu_function.labels.update(data_storage_tpu_labels)
    mobilenet_tflite_function.labels.update(data_storage_tflite_labels)

    deployment = FunctionDeployment(
        images.mobilenet_inference_function,
        {
            images.mobilenet_inference_tpu_manifest: mobilenet_tpu_function,
            images.mobilenet_inference_tflite_manifest: mobilenet_tflite_function,
        },
        ranking
    )

    deployment.function_factor = {
        images.mobilenet_inference_tpu_manifest: 1,
        images.mobilenet_inference_tflite_manifest: 1
    }

    return deployment


def get_resnet_training_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                                   ranking: DeploymentRanking) -> FunctionDeployment:
    resnet_training_gpu_characterization = FunctionCharacterization(images.resnet50_training_gpu_manifest,
                                                                    fet_oracle,
                                                                    resource_oracle)

    resnet_training_cpu_characterization = FunctionCharacterization(images.resnet50_training_cpu_manifest,
                                                                    fet_oracle,
                                                                    resource_oracle)
    data = storage.resnet_train_bucket_item.name

    data_storage_labels = {
        'data.skippy.io/receives-from-storage': '58M',
        'data.skippy.io/sends-to-storage': '103M',
        'data.skippy.io/receives-from-storage/path': f'{storage.resnet_train_bucket}/{data}',
        'data.skippy.io/sends-to-storage/path': f'{storage.resnet_train_bucket}/updated_model'
    }

    resnet_training_gpu_function = FunctionDefinition(
        name=images.resnet50_training_function,
        image=images.resnet50_training_gpu_manifest,
        characterization=resnet_training_gpu_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '2', 'device.edgerun.io/accelerator': 'GPU',
                'device.edgerun.io/vram': '2000', })

    resnet_training_gpu_function.labels.update(data_storage_labels)
    resnet_training_gpu_function.requests = Resources.from_str("800Mi", "1000m")

    resnet_training_cpu_function = FunctionDefinition(
        name=images.resnet50_training_function,
        image=images.resnet50_training_cpu_manifest,
        characterization=resnet_training_cpu_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '2'})

    resnet_training_cpu_function.labels.update(data_storage_labels)
    resnet_training_cpu_function.requests = Resources.from_str("1Gi", "1000m")
    deployment = FunctionDeployment(
        images.resnet50_training_function,
        {
            images.resnet50_training_gpu_manifest: resnet_training_gpu_function,
            images.resnet50_training_cpu_manifest: resnet_training_cpu_function
        },
        ranking
    )

    deployment.function_factor = {
        images.resnet50_training_gpu_manifest: 1,
        images.resnet50_training_cpu_manifest: 1
    }

    return deployment


def get_tf_gpu_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                          ranking: DeploymentRanking):
    tf_gpu_characterization = FunctionCharacterization(images.tf_gpu_manifest,
                                                       fet_oracle,
                                                       resource_oracle)

    tf_gpu_function = FunctionDefinition(
        name=images.tf_gpu_function,
        image=images.tf_gpu_manifest,
        characterization=tf_gpu_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '3', 'device.edgerun.io/accelerator': 'GPU',
                'device.edgerun.io/vram': '2000', })

    tf_gpu_function.requests = Resources.from_str("300Mi", "1000m")
    deployment = FunctionDeployment(
        images.tf_gpu_function,
        {
            images.tf_gpu_manifest: tf_gpu_function,
        },
        ranking
    )

    deployment.function_factor = {
        images.tf_gpu_manifest: 1
    }
    return deployment


def get_pi_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                      ranking: DeploymentRanking) -> FunctionDeployment:
    pi_characterization = FunctionCharacterization(images.pi_manifest,
                                                   fet_oracle,
                                                   resource_oracle)

    pi_function = FunctionDefinition(
        name=images.pi_function,
        image=images.pi_manifest,
        characterization=pi_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '1'})

    pi_function.requests = Resources.from_str("100Mi", "1000m")
    deployment = FunctionDeployment(
        images.pi_function,
        {
            images.pi_manifest: pi_function,
        },
        ranking
    )

    deployment.function_factor = {
        images.pi_manifest: 1
    }
    return deployment


def get_fio_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                       ranking: DeploymentRanking) -> FunctionDeployment:
    fio_characterization = FunctionCharacterization(images.fio_manifest,
                                                    fet_oracle,
                                                    resource_oracle)

    fio_function = FunctionDefinition(
        name=images.fio_function,
        image=images.fio_manifest,
        characterization=fio_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '1'})
    fio_function.requests = Resources.from_str("200Mi", "1000m")
    deployment = FunctionDeployment(
        images.fio_function,
        {
            images.fio_manifest: fio_function,
        },
        ranking
    )

    deployment.function_factor = {
        images.fio_manifest: 1
    }

    return deployment


def get_resnet_preprocessing_deployment(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                                        ranking: DeploymentRanking):
    resnet_preprocessing_characterization = FunctionCharacterization(images.resnet50_preprocessing_manifest,
                                                                     fet_oracle,
                                                                     resource_oracle)

    data = storage.resnet_pre_bucket_item.name
    data_storage_labels = {
        'data.skippy.io/receives-from-storage': '14M',
        'data.skippy.io/sends-to-storage': '14M',
        'data.skippy.io/receives-from-storage/path': f'{storage.resnet_pre_bucket}/{data}',
        'data.skippy.io/sends-to-storage/path': f'{storage.resnet_pre_bucket}/preprocessed'
    }

    resnet_preprocessing_function = FunctionDefinition(

        name=images.resnet50_preprocessing_function,
        image=images.resnet50_preprocessing_manifest,
        characterization=resnet_preprocessing_characterization,
        labels={'watchdog': 'http', 'workers': '4', 'cluster': '1'}
    )

    resnet_preprocessing_function.labels.update(data_storage_labels)
    resnet_preprocessing_function.requests = Resources.from_str("100Mi", "1000m")

    deployment = FunctionDeployment(
        images.resnet50_preprocessing_function,
        {
            images.resnet50_preprocessing_manifest: resnet_preprocessing_function,
        },
        ranking
    )

    deployment.function_factor = {
        images.resnet50_preprocessing_manifest: 1
    }

    return deployment


def create_all_deployments(fet_oracle: FetOracle, resource_oracle: ResourceOracle,
                           deployment_rankings: DeploymentSettings = None) -> Dict[str, FunctionDeployment]:
    if deployment_rankings is None:
        deployment_rankings = DeploymentSettings()
    return {
        images.resnet50_inference_function: get_resnet50_inference_deployment(fet_oracle, resource_oracle,
                                                                              deployment_rankings.resnet_inference_ranking),
        images.resnet50_training_function: get_resnet_training_deployment(fet_oracle, resource_oracle,
                                                                          deployment_rankings.resnet_training_ranking),
        images.mobilenet_inference_function: get_mobilenet_inference_deployment(fet_oracle, resource_oracle,
                                                                                deployment_rankings.mobilenet_inference_ranking),
        # images.fio_function: get_fio_deployment(fet_oracle, resource_oracle, deployment_rankings.fio_ranking),
        # images.pi_function: get_pi_deployment(fet_oracle, resource_oracle, deployment_rankings.pi_ranking),
        # images.tf_gpu_function: get_tf_gpu_deployment(fet_oracle, resource_oracle, deployment_rankings.tf_gpu_ranking),
        images.speech_inference_function: get_speech_inference_deployment(fet_oracle, resource_oracle,
                                                                          deployment_rankings.speech_inference_ranking),
        images.resnet50_preprocessing_function: get_resnet_preprocessing_deployment(fet_oracle, resource_oracle,
                                                                                    deployment_rankings.resnet_preprocessing_ranking)
    }
