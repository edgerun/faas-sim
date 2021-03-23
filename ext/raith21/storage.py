from ether.util import parse_size_string
from skippy.core.storage import DataItem

resnet_train_bucket = 'bucket_resnet50_train'
resnet_pre_bucket = 'bucket_resnet50_pre'
resnet_model_bucket = 'bucket_resnet50_model'
speech_bucket = 'bucket_speech'
mobilenet_bucket = 'mobilenet_bucket'

resnet_train_bucket_item = DataItem(resnet_train_bucket, 'raw_data', parse_size_string('58M'))
resnet_pre_bucket_item = DataItem(resnet_pre_bucket, 'raw_data', parse_size_string('14M'))
resnet_model_bucket_item = DataItem(resnet_model_bucket, 'model', parse_size_string('103M'))
speech_model_tflite_bucket_item = DataItem(speech_bucket, 'model_tflite', parse_size_string('48M'))
speech_model_gpu_bucket_item = DataItem(speech_bucket, 'model_gpu', parse_size_string('188M'))
mobilenet_model_tflite_bucket_item = DataItem(mobilenet_bucket, 'model_tflite', parse_size_string('4M'))
mobilenet_model_tpu_bucket_item = DataItem(mobilenet_bucket, 'model_tpu', parse_size_string('4M'))

bucket_names = [
    resnet_model_bucket,
    resnet_train_bucket,
    resnet_pre_bucket,
    mobilenet_bucket,
    speech_bucket
]

data_items = [
    resnet_train_bucket_item,
    resnet_pre_bucket_item,
    resnet_model_bucket_item,
    speech_model_gpu_bucket_item,
    speech_model_tflite_bucket_item,
    mobilenet_model_tpu_bucket_item,
    mobilenet_model_tflite_bucket_item
]
