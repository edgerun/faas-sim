from collections import Counter
from enum import Enum
from typing import List, Callable

import numpy as np

from .device import GpuDevice, Device
from .model import Arch, Accelerator, Bins, Location, Connection, Disk, Requirements


def count_attribute(devices: List[Device], values: List, getter: Callable[[Device], Enum]):
    counter = {}
    if len(devices) == 0:
        return {}
    for attr in values:
        counter[attr] = 0
    for device in devices:
        counter[getter(device)] += 1
    percentage = {}
    n_devices = len(devices)
    for attr, count in counter.items():
        percentage[attr] = count / n_devices
    return percentage


def get_gpu_model_count(devices: List[Device]):
    model_counts = Counter()
    gpu_mhz_counts = Counter()
    gpu_vram_counts = Counter()
    counter = 0
    for device in devices:
        if type(device) is GpuDevice:
            counter += 1
            gpu_device: GpuDevice
            gpu_device = device
            model_counts[gpu_device.gpu_model] += 1
            gpu_mhz_counts[gpu_device.gpu_mhz] += 1
            gpu_vram_counts[gpu_device.vram] += 1

    gpu_model_percentage = {}
    mhz_percentage = {}
    vram_percentage = {}
    for k, v in model_counts.items():
        gpu_model_percentage[k] = v / len(devices)

    for k, v in gpu_mhz_counts.items():
        mhz_percentage[k] = v / len(devices)

    for k, v in gpu_vram_counts.items():
        vram_percentage[k] = v / len(devices)

    return gpu_model_percentage, mhz_percentage, vram_percentage


def calculate_requirements(devices: List[Device]) -> Requirements:
    arch = count_attribute(devices, list(Arch), lambda d: d.arch)
    accelerator = count_attribute(devices, list(Accelerator), lambda d: d.accelerator)
    cores = count_attribute(devices, list(Bins), lambda d: d.cores)
    location = count_attribute(devices, list(Location), lambda d: d.location)
    connection = count_attribute(devices, list(Connection), lambda d: d.connection)
    network = count_attribute(devices, list(Bins), lambda d: d.network)
    cpu_mhz = count_attribute(devices, list(Bins), lambda d: d.cpu_mhz)
    cpu = count_attribute(devices, list(set([x.cpu for x in devices])), lambda d: d.cpu)
    ram = count_attribute(devices, list(Bins), lambda d: d.ram)
    disk = count_attribute(devices, list(Disk), lambda d: d.disk)
    gpu_model_percentage, gpu_mhz_percentage, vram_percentage = get_gpu_model_count(devices)
    return Requirements(
        arch=arch,
        accelerator=accelerator,
        cores=cores,
        location=location,
        connection=connection,
        network=network,
        cpu_mhz=cpu_mhz,
        cpu=cpu,
        ram=ram,
        disk=disk,
        gpu_model=gpu_model_percentage,
        gpu_vram=gpu_mhz_percentage,
        gpu_mhz=vram_percentage
    )


def calculate_heterogeneity(p: Requirements, q: Requirements) -> float:
    entropy_p = 0
    entropy_q = 0
    for (p_enum, p_characteristic), (q_enum, q_characteristic) in zip(p.characteristics, q.characteristics):
        for value in list(p_enum):
            default_val = 0.0000000000000000000001
            p_char = p_characteristic.get(value, default_val)
            if p_char == 0:
                p_char = default_val
            entropy_p += p_char * np.log(p_char)
            q_char = q_characteristic.get(value, default_val)
            if q_char == 0:
                q_char = default_val
            entropy_q += q_char * np.log(q_char)

    return entropy_p - entropy_q
