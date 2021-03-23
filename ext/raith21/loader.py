import logging
import os

import joblib
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_urls = {
    'eb-jetson-nano-01.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-jetson-nano-01.sav',
    'eb-jetson-nx-01.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-jetson-nx-01.sav',
    'eb-jetson-tx2-01.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-jetson-tx2-01.sav',
    'eb-nuc7.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-nuc7.sav',
    'eb-rockpi.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-rockpi.sav',
    'eb-rpi3-01.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-rpi3-01.sav',
    'eb-rpi4-01.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-rpi4-01.sav',
    'eb-xeongpu.sav': 'https://owncloud.tuwien.ac.at/index.php/s/zpkdevN5kV36ewC/download?path=%2F&files=eb-xeongpu.sav',
}


def load_model(model_file: str):
    d = os.path.dirname(model_file)
    if not os.path.isdir(d):
        logger.info('creating model folder %s', d)
        os.makedirs(d)

    if not os.path.isfile(model_file):
        file_name = os.path.basename(model_file)
        try:
            url = _urls[file_name]
        except KeyError:
            raise ValueError(f'could not download model file {file_name}, no remote url available')
        download_with_progress(url, model_file)

    return joblib.load(model_file)


def download_with_progress(url, target, block_size=2 ** 13):
    logger.info('downloading %s from %s', target, url)
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(target, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise IOError('error downloading file')
