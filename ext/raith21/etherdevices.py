from typing import List

from ether.blocks.nodes import create_node, counters, create_rpi3_node, create_tx2_node, create_nuc_node
from ether.core import Node

from ext.raith21.device import Device, GpuDevice
from ext.raith21.model import Location, Disk, Bins, Accelerator, Arch, Connection, CpuModel, GpuModel


def create_rockpi(name=None) -> Node:
    name = name if name is not None else 'rockpi_%d' % next(counters['rockpi'])

    return create_node(name=name,
                       cpus=6, arch='aarch64', mem='4G',
                       labels={
                           'ether.edgerun.io/type': 'sbc',
                           'ether.edgerun.io/model': 'rockpi'
                       })


def create_rpi4_node(name=None) -> Node:
    name = name if name is not None else 'rpi4_%d' % next(counters['rpi4'])

    return create_node(name=name,
                       arch='arm32v7',
                       cpus=4,
                       mem='1G',
                       labels={
                           'ether.edgerun.io/type': 'sbc',
                           'beta.kubernetes.io/arch': 'arm',
                           'locality.skippy.io/type': 'edge'
                       })


def create_coral(name=None) -> Node:
    name = name if name is not None else 'coral_%d' % next(counters['coral'])

    return create_node(name=name,
                       cpus=4, arch='aarch64', mem='1G',
                       labels={
                           'ether.edgerun.io/type': 'sbc',
                           'ether.edgerun.io/model': 'coral'
                       })


def create_xeongpu(name=None) -> Node:
    name = name if name is not None else 'xeongpu_%d' % next(counters['xeongpu'])

    return create_node(name=name,
                       cpus=4, arch='x86', mem='8167784Ki',
                       labels={
                           'ether.edgerun.io/type': 'vm',
                           'ether.edgerun.io/model': 'vm',
                           'device.edgerun.io/vram': '6Gi',
                           'ether.edgerun.io/capabilities/cuda': '10',
                           'ether.edgerun.io/capabilities/gpu': 'turing',
                       })


def create_xeoncpu(name=None) -> Node:
    name = name if name is not None else 'xeoncpu_%d' % next(counters['xeoncpu'])

    return create_node(name=name,
                       cpus=4, arch='x86', mem='8167784Ki',
                       labels={
                           'ether.edgerun.io/type': 'vm',
                           'ether.edgerun.io/model': 'vm',
                       })


def create_nano(name=None) -> Node:
    name = name if name is not None else 'nano_%d' % next(counters['nano'])

    return create_node(name=name,
                       cpus=4, arch='aarch64', mem='4047252Ki',
                       labels={
                           'ether.edgerun.io/type': 'embai',
                           'ether.edgerun.io/model': 'nvidia_jetson_nano',
                           'ether.edgerun.io/capabilities/cuda': '5.3',
                           'ether.edgerun.io/capabilities/gpu': 'maxwell',
                       })


def create_nx(name=None) -> Node:
    name = name if name is not None else 'nx_%d' % next(counters['nx'])

    return create_node(name=name,
                       cpus=6, arch='aarch64', mem='8047252Ki',
                       labels={
                           'ether.edgerun.io/type': 'embai',
                           'ether.edgerun.io/model': 'nvidia_jetson_nx',
                           'ether.edgerun.io/capabilities/cuda': '7.2',
                           'ether.edgerun.io/capabilities/gpu': 'volta',
                       })


def create_node_from_device(d: Device) -> Node:
    device = d.copy()

    def create():
        if device.arch is Arch.ARM32:
            cpu_cores = device.cores is Bins.MEDIUM or device.cores is Bins.HIGH or device.cores is Bins.VERY_HIGH
            cpu_mhz = device.cpu_mhz is Bins.HIGH or device.cpu_mhz is Bins.VERY_HIGH
            if cpu_mhz or cpu_cores:
                rpi4 = create_rpi4_node()
                device.cores = Bins.MEDIUM  # 4
                device.ram = Bins.LOW  # 1 GB
                device.cpu_mhz = Bins.LOW  # 1.5 GHz
                device.connection = Connection.MOBILE
                device.cpu = CpuModel.ARM
                device.network = Bins.LOW
                device.location = Location.EDGE
                return rpi4, device
            else:
                rpi3 = create_rpi3_node()
                device.cores = Bins.MEDIUM  # 4
                device.ram = Bins.LOW  # 1 GB
                device.cpu_mhz = Bins.LOW  # 1.4 GHz
                device.connection = Connection.MOBILE
                device.cpu = CpuModel.ARM
                device.network = Bins.LOW
                device.location = Location.EDGE
                return rpi3, device
        elif device.arch is Arch.AARCH64:
            if device.accelerator is Accelerator.GPU:
                return create_aarch64_gpu(device)
            elif device.accelerator is Accelerator.NONE:
                rockpi = create_rockpi()
                device.cores = Bins.MEDIUM  # 6
                device.ram = Bins.LOW  # 2 GB
                device.cpu_mhz = Bins.MEDIUM  # 4x 1.4, 2x 1.8
                device.connection = Connection.MOBILE
                device.cpu = CpuModel.ARM
                device.network = Bins.LOW
                device.location = Location.EDGE
                device.disk = Disk.SD
                return rockpi, device
            else:
                coral = create_coral()
                device.location = Location.EDGE
                device.disk = Disk.FLASH
                device.cores = Bins.MEDIUM  # 4
                device.ram = Bins.LOW  # 1 GB
                device.cpu_mhz = Bins.LOW  # 1.5GHz
                device.connection = Connection.MOBILE
                device.cpu = CpuModel.ARM
                device.network = Bins.LOW
                return coral, device
        else:
            if device.location is not Location.CLOUD:
                nuc = create_nuc_node()
                copy = Device(
                    arch=Arch.X86,
                    id=device.id,
                    cores=Bins.MEDIUM,  # 4
                    ram=Bins.HIGH,  # 16
                    cpu_mhz=Bins.MEDIUM,  # 2.2GHz
                    connection=Connection.MOBILE,
                    cpu=CpuModel.I7,
                    network=Bins.LOW,  # due to mobilentwork in urban sensing
                    accelerator=Accelerator.NONE,
                    disk=Disk.NVME,
                    location=Location.EDGE
                )

                return nuc, copy
            else:
                if device.accelerator is Accelerator.GPU:
                    vm = create_xeongpu()
                    copy = GpuDevice(
                        id=device.id,
                        arch=Arch.X86,
                        accelerator=Accelerator.GPU,
                        cores=Bins.MEDIUM,  # 4
                        location=Location.CLOUD,
                        connection=Connection.ETHERNET,
                        network=Bins.HIGH,
                        cpu_mhz=Bins.HIGH,  # 3.44 GHz
                        cpu=CpuModel.XEON,
                        ram=Bins.MEDIUM,  # 4
                        vram=Bins.MEDIUM,  # 6 GB
                        gpu_mhz=Bins.VERY_HIGH,  # base: 1500 MHz, boost: 1770 MHz
                        gpu_model=GpuModel.TURING,
                        disk=Disk.SSD
                    )

                    vm.labels['device.edgerun.io/vram'] = '6000'
                else:
                    vm = create_xeoncpu()
                    copy = Device(
                        id=device.id,
                        arch=Arch.X86,
                        accelerator=Accelerator.NONE,
                        cores=Bins.MEDIUM,  # 4
                        location=Location.CLOUD,
                        connection=Connection.ETHERNET,
                        network=Bins.HIGH,
                        cpu_mhz=Bins.HIGH,  # 3.44 GHz
                        cpu=CpuModel.XEON,
                        ram=Bins.MEDIUM,  # 8 GB
                        disk=Disk.SSD
                    )
                return vm, copy

    node, device = create()

    node.labels.update(device.labels)
    if device.location is Location.CLOUD:
        node.labels['locality.skippy.io/type'] = 'cloud'
    else:
        node.labels['locality.skippy.io/type'] = 'edge'
    if device.accelerator is Accelerator.GPU:
        node.labels['capability.skippy.io/nvidia-cuda'] = '10'
        node.labels['capability.skippy.io/nvidia-gpu'] = ''
    return node


def create_aarch64_gpu(device):
    adevice: GpuDevice
    adevice = device
    if device.ram is Bins.LOW or device.cpu_mhz is Bins.LOW or adevice.gpu_model is GpuModel.MAXWELL:
        node = create_nano()
        device.gpu_model = GpuModel.MAXWELL
        device.gpu_mhz = Bins.LOW  # base: 640 MHz, boost: 921 MHz
        device.location = Location.EDGE
        device.disk = Disk.SD
        device.cores = Bins.MEDIUM  # 4
        device.network = Bins.LOW
        device.connection = Connection.MOBILE
        device.cpu = CpuModel.ARM
        device.cpu_mhz = Bins.LOW  # 1.43 GHz
        device.vram = Bins.LOW  # 4 GB shared
        device.ram = Bins.LOW  # 4 GB
        node.labels['device.edgerun.io/vram'] = '4000'

    elif device.gpu_model is GpuModel.PASCAL or device.cores is Bins.LOW:
        tx2_device: GpuDevice
        tx2_device = device
        node = create_tx2_node()
        tx2_device.cores = Bins.MEDIUM  # 4
        tx2_device.ram = Bins.MEDIUM  # 8 GB
        tx2_device.disk = Disk.FLASH
        tx2_device.cpu_mhz = Bins.MEDIUM  # 2 Ghz
        tx2_device.connection = Connection.MOBILE
        tx2_device.cpu = CpuModel.ARM
        device.location = Location.EDGE
        tx2_device.network = Bins.LOW  # mobile in urban
        tx2_device.vram = Bins.MEDIUM  # 8 GB /shared with ram
        tx2_device.gpu_mhz = Bins.HIGH  # base: 854 MHz, boost: 1464 MHz
        tx2_device.gpu_model = GpuModel.PASCAL
        node.labels['device.edgerun.io/vram'] = '8000'

    else:
        node = create_nx()
        device.cores = Bins.MEDIUM  # 6
        device.ram = Bins.MEDIUM  # 8 GB
        device.cpu_mhz = Bins.MEDIUM  # 1.9 GHz
        device.connection = Connection.MOBILE
        device.cpu = CpuModel.ARM
        device.location = Location.EDGE
        device.disk = Disk.SD
        device.network = Bins.LOW
        device.vram = Bins.MEDIUM  # 8 GB shared
        device.gpu_mhz = Bins.HIGH  # base 854 MHz, boost: 1377 MHz
        device.gpu_model = GpuModel.TURING
        node.labels['device.edgerun.io/vram'] = '8000'

    device.vram = Bins.LOW

    return node, device


def convert_to_ether_nodes(devices: List[Device]) -> List[Node]:
    nodes = []
    for index, device in enumerate(devices):
        nodes.append(create_node_from_device(device))

    return nodes


def create_device_from_node(node: Node):
    accelerator = Accelerator[node.labels['device.edgerun.io/accelerator']]
    if accelerator is Accelerator.GPU:
        return GpuDevice(
            id=node.name[:node.name.rindex('_')],
            arch=Arch[node.labels['device.edgerun.io/arch']],
            accelerator=accelerator,
            cores=Bins[node.labels['device.edgerun.io/cores']],
            location=Location[node.labels['device.edgerun.io/location']],
            connection=Connection[node.labels['device.edgerun.io/connection']],
            network=Bins[node.labels['device.edgerun.io/network']],
            cpu_mhz=Bins[node.labels['device.edgerun.io/cpu_mhz']],
            cpu=CpuModel[node.labels['device.edgerun.io/cpu']],
            ram=Bins[node.labels['device.edgerun.io/ram']],
            vram=Bins[node.labels['device.edgerun.io/vram_bin']],
            gpu_mhz=Bins[node.labels['device.edgerun.io/gpu_mhz']],
            gpu_model=GpuModel[node.labels['device.edgerun.io/gpu_model']],
            disk=Disk[node.labels['device.edgerun.io/disk']]
        )
    else:
        return Device(
            # FIXME this will be problematic when same nodes (i.e. same name) have diff. capabilities
            id=node.name[:node.name.rindex('_')],
            arch=Arch[node.labels['device.edgerun.io/arch']],
            accelerator=accelerator,
            cores=Bins[node.labels['device.edgerun.io/cores']],
            location=Location[node.labels['device.edgerun.io/location']],
            connection=Connection[node.labels['device.edgerun.io/connection']],
            network=Bins[node.labels['device.edgerun.io/network']],
            cpu_mhz=Bins[node.labels['device.edgerun.io/cpu_mhz']],
            cpu=CpuModel[node.labels['device.edgerun.io/cpu']],
            ram=Bins[node.labels['device.edgerun.io/ram']],
            disk=Disk[node.labels['device.edgerun.io/disk']]
        )


def convert_to_devices(nodes: List[Node]):
    return list(map(lambda n: create_device_from_node(n), nodes))
