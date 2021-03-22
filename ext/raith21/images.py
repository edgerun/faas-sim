resnet50_inference_cpu_manifest = 'faas-workloads/resnet-inference-cpu'
resnet50_inference_gpu_manifest = 'faas-workloads/resnet-inference-gpu'
resnet50_inference_function = 'resnet50-inference'

resnet50_training_gpu_manifest = 'faas-workloads/resnet-training-gpu'
resnet50_training_cpu_manifest = 'faas-workloads/resnet-training-cpu'
resnet50_training_function = 'resnet50-training'

speech_inference_tflite_manifest = 'faas-workloads/speech-inference-tflite'
speech_inference_gpu_manifest = 'faas-workloads/speech-inference-gpu'
speech_inference_function = 'speech-inference'

mobilenet_inference_tflite_manifest = 'faas-workloads/mobilenet-inference-tflite'
mobilenet_inference_tpu_manifest = 'faas-workloads/mobilenet-inference-tpu'
mobilenet_inference_function = 'mobilenet-inference'

resnet50_preprocessing_function = 'resnet50-preprocessing'
resnet50_preprocessing_manifest = 'faas-workloads/resnet-preprocessing'

pi_manifest = 'faas-workloads/python-pi'
pi_function = 'python-pi'

fio_manifest = 'faas-workloads/fio'
fio_function = 'fio'

tf_gpu_manifest = 'faas-workloads/tf-gpu'
tf_gpu_function = 'tf-gpu'
all_ai_images = [
    (resnet50_inference_cpu_manifest, '2000M', 'x86'),
    (resnet50_inference_cpu_manifest, '2000M', 'amd64'),
    (resnet50_inference_cpu_manifest, '700M', 'arm32v7'),
    (resnet50_inference_cpu_manifest, '700M', 'arm32'),
    (resnet50_inference_cpu_manifest, '700M', 'arm'),
    (resnet50_inference_cpu_manifest, '840M', 'aarch64'),
    (resnet50_inference_cpu_manifest, '840M', 'arm64'),

    (resnet50_inference_gpu_manifest, '2000M', 'x86'),
    (resnet50_inference_gpu_manifest, '2000M', 'amd64'),
    (resnet50_inference_gpu_manifest, '1000M', 'aarch64'),
    (resnet50_inference_gpu_manifest, '1000M', 'arm64'),

    (resnet50_training_gpu_manifest, '2000M', 'x86'),
    (resnet50_training_gpu_manifest, '2000M', 'amd64'),
    (resnet50_training_gpu_manifest, '1000M', 'aarch64'),
    (resnet50_training_gpu_manifest, '1000M', 'arm64'),

    (resnet50_training_cpu_manifest, '2000M', 'amd64'),
    (resnet50_training_cpu_manifest, '2000M', 'x86'),

    (speech_inference_tflite_manifest, '108M', 'amd64'),
    (speech_inference_tflite_manifest, '108M', 'x86'),
    (speech_inference_tflite_manifest, '328M', 'arm32v7'),
    (speech_inference_tflite_manifest, '328M', 'arm32'),
    (speech_inference_tflite_manifest, '328M', 'arm'),
    (speech_inference_tflite_manifest, '400M', 'arm64'),
    (speech_inference_tflite_manifest, '400M', 'aarch64'),

    (speech_inference_gpu_manifest, '1600M', 'amd64'),
    (speech_inference_gpu_manifest, '1600M', 'x86'),
    (speech_inference_gpu_manifest, '1300M', 'arm64'),
    (speech_inference_gpu_manifest, '1300M', 'aarch64'),

    (mobilenet_inference_tflite_manifest, '180M', 'amd64'),
    (mobilenet_inference_tflite_manifest, '180M', 'x86'),
    (mobilenet_inference_tflite_manifest, '160M', 'arm32v7'),
    (mobilenet_inference_tflite_manifest, '160M', 'arm32'),
    (mobilenet_inference_tflite_manifest, '160M', 'arm'),
    (mobilenet_inference_tflite_manifest, '173M', 'arm64'),
    (mobilenet_inference_tflite_manifest, '173M', 'aarch64'),

    (mobilenet_inference_tpu_manifest, '173M', 'arm64'),
    (mobilenet_inference_tpu_manifest, '173M', 'aarch64'),

    (pi_manifest, '88M', 'amd64'),
    (pi_manifest, '88M', 'x86'),
    (pi_manifest, '55M', 'arm32v7'),
    (pi_manifest, '55M', 'arm32'),
    (pi_manifest, '55M', 'arm'),
    (pi_manifest, '62M', 'arm64'),
    (pi_manifest, '62M', 'aarch64'),

    (fio_manifest, '24M', 'amd64'),
    (fio_manifest, '24M', 'x86'),
    (fio_manifest, '20M', 'arm32v7'),
    (fio_manifest, '20M', 'arm32'),
    (fio_manifest, '20M', 'arm'),
    (fio_manifest, '23M', 'arm64'),
    (fio_manifest, '23M', 'aarch64'),

    (tf_gpu_manifest, '4100M', 'amd64'),
    (tf_gpu_manifest, '4100M', 'x86'),
    (tf_gpu_manifest, '2240M', 'arm64'),
    (tf_gpu_manifest, '2240M', 'aarch64'),

    (resnet50_preprocessing_manifest, '4100M', 'x86'),
    (resnet50_preprocessing_manifest, '4100M', 'amd64'),
    (resnet50_preprocessing_manifest, '1370M', 'arm32v7'),
    (resnet50_preprocessing_manifest, '1370M', 'arm32'),
    (resnet50_preprocessing_manifest, '1370M', 'arm'),
    (resnet50_preprocessing_manifest, '1910M', 'arm64'),
    (resnet50_preprocessing_manifest, '1910M', 'aarch64')
]
