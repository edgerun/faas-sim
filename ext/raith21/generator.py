import datetime
import multiprocessing
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing.context import Process
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import itertools
import numpy as np

from ext.raith21.device import ArchProperties, GpuDevice, Device
from ext.raith21.model import Arch, Requirements, Accelerator, Bins, Disk, Location, Connection, CpuModel, GpuModel


@dataclass
class GeneratorSettings:
    arch: Dict[Arch, float]
    properties: Dict[Arch, ArchProperties]


def xeon_reqs():
    xeon_single_device_req = Requirements(
        arch={
            Arch.X86: 1
        },
        accelerator={
            Accelerator.NONE: 1
        },
        cores={
            Bins.LOW: 1
        },
        disk={
            Disk.SSD: 1
        },
        location={
            Location.CLOUD: 1
        },
        connection={
            Connection.ETHERNET: 1
        },
        network={
            Bins.LOW: 1
        },
        cpu_mhz={
            Bins.LOW: 1
        },
        cpu={
            CpuModel.XEON: 1
        },
        ram={
            Bins.LOW: 1
        },
        gpu_model={},
        gpu_vram={},
        gpu_mhz={}
    )
    return xeon_single_device_req


def create_generator(arches, t, heterogeneity_score, base_req, folder):
    gen_settings = {}
    arch_settings = {}
    for index, arch in enumerate(arches):
        arch_settings[arch[0]] = arch[1]
        settings = list(map(lambda i: create_t_setting(i, t[index]), range(len(t[index]))))
        settings = [arch] + settings
        gen_settings[arch[0]] = ArchProperties(*settings)
    setting = GeneratorSettings(arch_settings, gen_settings)
    save_setting(folder, setting)
    return setting


def create_t_setting(i, t):
    setting = {}
    for k, prob in t[i]:
        setting[k] = prob
    return setting


def create_settings(arches, base_req, tuples, heterogeneity_score: Callable[[Requirements, Requirements], float]
                    , folder: str):
    product = itertools.product(*tuples)
    list((map(lambda t: create_generator(arches, t, heterogeneity_score, base_req, folder), product)))


def create_and_save_settings(arches, base_requirement, tuples, heterogeneity_score, folder):
    combs = list(tuples.values())
    create_settings(arches, base_requirement, combs, heterogeneity_score, folder)


def save_setting(folder, setting):
    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S_%f')
    file_name = f'{now}_{random.randint(1000, 10000)}.pickle'
    with open(f'{folder}/{file_name}', 'wb+') as fd:
        pickle.dump(setting, fd)


def choose_attribute_settings(values, percentage):
    if type(values) is tuple:
        return np.array([values])
    n = len(values)
    take = max(1, int(n * percentage))
    choice = np.random.choice(n, size=take)
    return np.array(values)[choice, :]


def process_arches(arch_probs, probs_for_archs, base_req, heterogeneity_score, folder):
    for arches in arch_probs:
        tuples = {}
        for arch in list(Arch):
            arch_tuples = list(itertools.product(*probs_for_archs[arch]))
            tuples[arch] = arch_tuples
        create_and_save_settings(
            arches,
            base_req,
            tuples,
            heterogeneity_score,
            folder
        )


def filter_invalid_settings(old_probs_per_arch):
    # no gpu/tpu arm32
    old_probs_per_arch[Arch.ARM32]['accelerator'] = ((Accelerator.NONE, 1), (Accelerator.GPU, 0), (Accelerator.TPU, 0))
    del old_probs_per_arch[Arch.ARM32]['gpu_vram']
    del old_probs_per_arch[Arch.ARM32]['gpu_model']
    del old_probs_per_arch[Arch.ARM32]['gpu_mhz']
    old_probs_per_arch[Arch.ARM32]['cpu'] = ((CpuModel.I7, 0), (CpuModel.XEON, 0), (CpuModel.ARM, 1))
    old_probs_per_arch[Arch.AARCH64]['cpu'] = ((CpuModel.I7, 0), (CpuModel.XEON, 0), (CpuModel.ARM, 1))

    old_probs_per_arch[Arch.ARM32]['disk'] = (
        (Disk.SD, 1), (Disk.FLASH, 0), (Disk.HDD, 0), (Disk.SSD, 0), (Disk.NVME, 0))

    # filter very high cores
    old_probs_per_arch[Arch.ARM32]['cores'] = list(
        filter(lambda f: f[3][1] == 0, old_probs_per_arch[Arch.ARM32]['cores']))
    # filter very high ram
    old_probs_per_arch[Arch.ARM32]['ram'] = list(filter(lambda f: f[3][1] == 0, old_probs_per_arch[Arch.ARM32]['ram']))
    # filter high ram
    old_probs_per_arch[Arch.ARM32]['ram'] = list(filter(lambda f: f[2][1] == 0, old_probs_per_arch[Arch.ARM32]['ram']))
    # filter high cores
    old_probs_per_arch[Arch.ARM32]['cores'] = list(
        filter(lambda f: f[2][1] == 0, old_probs_per_arch[Arch.ARM32]['cores']))
    # filter cloud
    old_probs_per_arch[Arch.ARM32]['location'] = list(
        filter(lambda f: f[0][1] == 0, old_probs_per_arch[Arch.ARM32]['location']))
    # filter mec
    old_probs_per_arch[Arch.ARM32]['location'] = list(
        filter(lambda f: f[2][1] == 0, old_probs_per_arch[Arch.ARM32]['location']))
    # filter all tpu x86
    old_probs_per_arch[Arch.X86]['accelerator'] = list(
        filter(lambda f: f[2][1] == 0, old_probs_per_arch[Arch.X86]['accelerator']))
    # filter mobile connection
    old_probs_per_arch[Arch.X86]['connection'] = list(
        filter(lambda f: f[0][1] == 0, old_probs_per_arch[Arch.X86]['connection']))
    # filter wifi connection
    old_probs_per_arch[Arch.X86]['connection'] = list(
        filter(lambda f: f[1][1] == 0, old_probs_per_arch[Arch.X86]['connection']))
    # filter arm cpu
    old_probs_per_arch[Arch.X86]['cpu'] = list(filter(lambda f: f[2][1] == 0, old_probs_per_arch[Arch.X86]['cpu']))
    # filter maxwell
    old_probs_per_arch[Arch.X86]['gpu_model'] = list(
        filter(lambda f: f[1][1] == 0, old_probs_per_arch[Arch.X86]['gpu_model']))
    # filter pascal
    old_probs_per_arch[Arch.X86]['gpu_model'] = list(
        filter(lambda f: f[2][1] == 0, old_probs_per_arch[Arch.X86]['gpu_model']))
    # filter volta
    old_probs_per_arch[Arch.X86]['gpu_model'] = list(
        filter(lambda f: f[3][1] == 0, old_probs_per_arch[Arch.X86]['gpu_model']))
    # filter low network
    old_probs_per_arch[Arch.X86]['network'] = list(
        filter(lambda f: f[0][1] == 0, old_probs_per_arch[Arch.X86]['network']))
    # filter hdd disk
    old_probs_per_arch[Arch.X86]['disk'] = list(
        filter(lambda f: f[0][1] == 0, old_probs_per_arch[Arch.X86]['disk']))
    # filter flash disk
    old_probs_per_arch[Arch.X86]['disk'] = list(
        filter(lambda f: f[3][1] == 0, old_probs_per_arch[Arch.X86]['disk']))
    # filter sd disk
    old_probs_per_arch[Arch.X86]['disk'] = list(
        filter(lambda f: f[4][1] == 0, old_probs_per_arch[Arch.X86]['disk']))

    # filter very high cores
    old_probs_per_arch[Arch.AARCH64]['cores'] = list(
        filter(lambda f: f[3][1] == 0, old_probs_per_arch[Arch.AARCH64]['cores']))
    # filter hdd disk
    old_probs_per_arch[Arch.AARCH64]['disk'] = list(
        filter(lambda f: f[0][1] == 0, old_probs_per_arch[Arch.AARCH64]['disk']))
    # filter ssd disk
    old_probs_per_arch[Arch.AARCH64]['disk'] = list(
        filter(lambda f: f[1][1] == 0, old_probs_per_arch[Arch.AARCH64]['disk']))
    # filter nvme disk
    # TODO remove this in case with get nvme for xavier
    old_probs_per_arch[Arch.AARCH64]['disk'] = list(
        filter(lambda f: f[3][1] == 0, old_probs_per_arch[Arch.AARCH64]['disk']))
    # filter cloud
    old_probs_per_arch[Arch.AARCH64]['location'] = list(
        filter(lambda f: f[0][1] == 0, old_probs_per_arch[Arch.AARCH64]['location']))
    # filter turing
    old_probs_per_arch[Arch.AARCH64]['gpu_model'] = list(
        filter(lambda f: f[0][1] == 0, old_probs_per_arch[Arch.AARCH64]['gpu_model']))
    return old_probs_per_arch


def generate_settings(base_requirement: Requirements,
                      heterogeneity_score: Callable[[Requirements, Requirements], float], steps: int = 5,
                      arch_steps: int = 5,
                      percentage: float = 1,
                      folder: str = './data', cores: int = None) -> None:
    """Saves the settings in the given folder"""
    probs_for_archs = generate_arch_probs(arch_steps)
    old_probs_per_arch = generate_probabilities(steps)
    if cores is None:
        cores = multiprocessing.cpu_count()

    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S')
    folder = f'{folder}/{now}_archsteps-{arch_steps}_steps-{steps}_percentage-{percentage}'
    Path(folder).mkdir(parents=True, exist_ok=True)
    old_probs_per_arch = filter_invalid_settings(old_probs_per_arch)
    probs_per_arch = defaultdict(list)
    for arch, old_probs in old_probs_per_arch.items():
        for values in old_probs.values():
            probs_per_arch[arch].append(choose_attribute_settings(values, percentage))

    split = np.array_split(probs_for_archs, cores)
    ps = []
    for i in range(cores):
        if len(split[i]) > 0:
            # process_arches(split[i], probs_per_arch, base_requirement, heterogeneity_score, folder,)
            p = Process(target=process_arches,
                        args=(split[i], probs_per_arch, base_requirement, heterogeneity_score, folder,))
            p.start()
            ps.append(p)

    for p in ps:
        p.join()


def generate_probabilities(steps: int):
    # TODO update this to ArchProperties, i.e.: create for each arch probabilites for the properties
    space = np.linspace(0, 1, num=steps)
    probs = defaultdict(lambda: defaultdict(list))
    for name, enum in Requirements.fields():
        if name == 'arch':
            continue
        values = list(enum)
        for t in itertools.product(space, repeat=len(values)):
            if np.sum(t) == 1:
                print(t)
                for index, prob in enumerate(t):
                    probs[name][values[index]].append(prob)
    tupled_probs = defaultdict(list)
    for key, value in probs.items():
        keys = list(value.keys())
        first_key = keys[0]
        for i in range(len(value[first_key])):
            l = []
            for k in keys:
                l.append((k, value[k][i]))
            tupled_probs[key].append(tuple(l))

    probs = {}
    for arch in list(Arch):
        probs[arch] = tupled_probs.copy()
    return probs


def generate_arch_probs(arch_steps: int):
    space = np.linspace(0, 1, num=arch_steps)
    probs = defaultdict(lambda: defaultdict(list))

    values = list(Arch)
    for t in itertools.product(space, repeat=len(values)):
        if np.sum(t) == 1:
            print(t)
            has_one = False
            for val in t:
                if val == 1:
                    has_one = True
                    break
            if not has_one:
                for index, prob in enumerate(t):
                    probs['arch'][values[index]].append(prob)
    tupled_probs = defaultdict(list)
    for key, value in probs.items():
        keys = list(value.keys())
        first_key = keys[0]
        for i in range(len(value[first_key])):
            l = []
            for k in keys:
                l.append((k, value[k][i]))
            tupled_probs[key].append(tuple(l))

    return tupled_probs['arch']


def random_network_throughput(bin: Bins) -> Tuple[int, int]:
    if bin is Bins.LOW:
        return 125, 25
    if bin is Bins.MEDIUM:
        return 250, 50
    if bin is Bins.HIGH:
        return 500, 500
    if bin is Bins.VERY_HIGH:
        return 100, 100


def random_ram_size(bin: Bins) -> int:
    if bin is Bins.LOW:
        return random.choice([1, 2, 4])
    if bin is Bins.MEDIUM:
        return random.choice([8, 16, 32])
    if bin is Bins.HIGH:
        return random.choice([64, 128])
    if bin is Bins.VERY_HIGH:
        return random.choice([256])


def random_cpu_cores(bin: Bins) -> int:
    if bin is Bins.LOW:
        return random.randint(1, 2)
    elif bin is Bins.MEDIUM:
        return random.choice([4, 6, 8, 12])
    elif bin is Bins.HIGH:
        return random.choice([16, 32])
    elif bin is Bins.VERY_HIGH:
        return random.choice([64, 88, 128])


def create_tuples(probs, name, enum):
    values = list(enum)
    return [tuple(values)] * len(probs[name][values[0]])


def random_arch():
    return random.choice(list(Arch))


def random_bin():
    return random.choice(list(Bins))


def random_connection():
    return random.choice(list(Connection))


def random_location():
    return random.choice(list(Location))


def random_cpu(arch: Arch) -> CpuModel:
    if arch is Arch.AARCH64:
        return CpuModel.ARM
    elif arch is Arch.ARM32:
        return CpuModel.ARM
    else:
        return random.choice([CpuModel.I7, CpuModel.XEON])


def random_accelerator(arch: Arch) -> Accelerator:
    if arch is Arch.AARCH64:
        return random.choice(list(Accelerator))
    elif arch is Arch.ARM32:
        return Accelerator.NONE
    else:
        return random.choice([Accelerator.GPU, Accelerator.NONE])


def random_gpu_model(arch: Arch) -> GpuModel:
    if arch is Arch.X86:
        return GpuModel.TURING
    else:
        return random.choice([GpuModel.PASCAL, GpuModel.MAXWELL, GpuModel.VOLTA])


def random_disk() -> Disk:
    return random.choice(list(Disk))


def get_property_with_probs(probs: Dict[Enum, float]):
    values = list(probs[1].keys())
    probs = list(probs[1].values())
    if len(values) == 0:
        return None
    index = np.random.choice(len(values), size=1, p=probs)[0]
    return values[index]


def generate_devices_with_settings(n: int, settings: GeneratorSettings) -> List[Device]:
    devices = []
    device_id = 0
    for arch, proportion in settings.arch.items():
        for i in range(int(n * proportion)):
            characteristics = list(map(lambda x: get_property_with_probs(x), settings.properties[arch].values))
            if characteristics[0] is Accelerator.GPU:
                device = GpuDevice(str(device_id), arch, *characteristics)
            else:
                # remove gpu characteristics with slice
                device = Device(str(device_id), arch, *characteristics[:9])
            devices.append(device)
    return devices


def generate_devices(n: int, settings: GeneratorSettings = None) -> List[Device]:
    if settings is not None:
        return generate_devices_with_settings(n, settings)
    devices = []
    for i in range(n):
        device_id = str(i)
        arch = random_arch()
        cores = random_bin()
        location = random_location()
        connection = random_connection()
        network = random_bin()
        cpu_mhz = random_bin()
        cpu = random_cpu(arch)
        disk = random_disk()
        ram = random_bin()
        accelerator = random_accelerator(arch)
        if accelerator is Accelerator.GPU:
            vram = random_bin()
            gpu_mhz = random_bin()
            gpu_model = random_gpu_model(arch)
            devices.append(GpuDevice(
                id=device_id,
                arch=arch,
                accelerator=accelerator,
                cores=cores,
                location=location,
                connection=connection,
                network=network,
                cpu_mhz=cpu_mhz,
                cpu=cpu,
                ram=ram,
                vram=vram,
                gpu_mhz=gpu_mhz,
                gpu_model=gpu_model,
                disk=disk
            ))
        else:
            devices.append(Device(
                id=device_id,
                arch=arch,
                accelerator=accelerator,
                cores=cores,
                location=location,
                connection=connection,
                network=network,
                cpu_mhz=cpu_mhz,
                cpu=cpu,
                ram=ram,
                disk=disk
            ))

    return devices


def main():
    generate_settings_main()


def generate_settings_main():
    # steps = 3
    steps = 4
    # arch_steps = 3
    arch_steps = 6
    # percentage = 0.4
    percentage = 0.1
    folder = '/mnt/ssd2data/Documents/hw_mapping_gen_settings'
    cores = 4
    base_req = xeon_reqs()
    score_f = calculate_diff_entropy
    generate_settings(
        base_requirement=base_req,
        heterogeneity_score=score_f,
        steps=steps,
        arch_steps=arch_steps,
        percentage=percentage,
        folder=folder,
        cores=cores
    )


if __name__ == '__main__':
    main()
