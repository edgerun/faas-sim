from dataclasses import dataclass
from enum import Enum
from typing import Dict

"""
Bins:    |    LOW     |    MEDIUM    |     HIGH     | VERY_HIGH

Cores:   | 1-2        |   4 - 8      |  16 - 32     | > 32
RAM:     | 1-2        |   4 - 8      |  16 - 32     | > 32
CpuMhz:  | <= 1.5     |   1.6 - 2.2  |     < 3.5    | > 3.5
GpuMHz:  | <= 1000    |   <= 1200    |  <= 1500     | > 1700
VRAM:    | <= 2       |   4 - 8      |   < 32       | > 32    
Network: | <= 150Mbps | <= 500 Mbps  | <=1 Gbit     | >= 10 Gbit 
"""

class Bins(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class Location(Enum):
    CLOUD = 1
    EDGE = 2
    MEC = 3
    MOBILE = 4


class Disk(Enum):
    HDD = 1
    SSD = 2
    NVME = 3
    FLASH = 4
    SD = 5


class Accelerator(Enum):
    NONE = 1
    GPU = 2
    TPU = 3


class Connection(Enum):
    MOBILE = 1
    WIFI = 2
    ETHERNET = 3


class Arch(Enum):
    ARM32 = 1
    X86 = 2
    AARCH64 = 3


class GpuModel(Enum):
    TURING = 1
    PASCAL = 2
    MAXWELL = 3
    VOLTA = 4


class CpuModel(Enum):
    I7 = 1
    XEON = 2
    ARM = 3


@dataclass
class Requirements:
    arch: Dict[Arch, float]
    accelerator: Dict[Accelerator, float]
    cores: Dict[Bins, float]
    disk: Dict[Disk, float]
    location: Dict[Location, float]
    connection: Dict[Connection, float]
    network: Dict[Bins, float]
    cpu_mhz: Dict[Bins, float]
    cpu: Dict[CpuModel, float]
    ram: Dict[Bins, float]
    gpu_vram: Dict[Bins, float]
    gpu_mhz: Dict[Bins, float]
    gpu_model: Dict[GpuModel, float]

    def __str__(self):
        def join(d: Dict) -> str:
            return "\n".join(['%s:: %s' % (key, value) for (key, value) in d.items()])

        text = "---------------------------"
        for name, c in self.characteristics:
            text += f'\n--------{name}---------\n'
            text += join(c)
            text += '\n'
        return text

    def __map(self, d: Dict[Enum, float]) -> Dict[str, float]:
        data = {}
        for k, v in d.items():
            data[f'{str(k.name)}'] = v
        return data

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            'device.edgerun.io/arch': self.__map(self.arch),
            'device.edgerun.io/accelerator': self.__map(self.accelerator),
            'device.edgerun.io/cores': self.__map(self.cores),
            'device.edgerun.io/disk': self.__map(self.disk),
            'device.edgerun.io/location': self.__map(self.location),
            'device.edgerun.io/connection': self.__map(self.connection),
            'device.edgerun.io/network': self.__map(self.network),
            'device.edgerun.io/cpu_mhz': self.__map(self.cpu_mhz),
            'device.edgerun.io/cpu': self.__map(self.cpu),
            'device.edgerun.io/ram': self.__map(self.ram),
            'device.edgerun.io/vram_bin': self.__map(self.gpu_vram),
            'device.edgerun.io/gpu_mhz': self.__map(self.gpu_mhz),
            'device.edgerun.io/gpu_model': self.__map(self.gpu_model),
        }

    @property
    def characteristics(self):
        return [
            (Arch, self.arch),
            (Accelerator, self.accelerator),
            (Bins, self.cores),
            (Disk, self.disk),
            (Location, self.location),
            (Connection, self.connection),
            (Bins, self.network),
            (Bins, self.cpu_mhz),
            (Bins, self.cpu),
            (Bins, self.ram),
            (Bins, self.gpu_vram),
            (Bins, self.gpu_mhz),
            (GpuModel, self.gpu_model)
        ]

    @staticmethod
    def fields():
        return [
            ('arch', Arch),
            ('accelerator', Accelerator),
            ('cores', Bins),
            ('disk', Disk),
            ('location', Location),
            ('connection', Connection),
            ('network', Bins),
            ('cpu_mhz', Bins),
            ('cpu', CpuModel),
            ('ram', Bins),
            ('gpu_vram', Bins),
            ('gpu_mhz', Bins),
            ('gpu_model', GpuModel)
        ]
