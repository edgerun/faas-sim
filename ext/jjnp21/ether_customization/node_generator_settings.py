
# todo potentially reset this
from ext.raith21.device import ArchProperties
from ext.raith21.generator import GeneratorSettings
from ext.raith21.model import *

edge_intelligence_settings = GeneratorSettings(
    arch={
        Arch.X86: 0,
        Arch.AARCH64: 0.7,
        Arch.ARM32: 0.3
    },
    properties={
        Arch.X86: ArchProperties(
            arch=Arch.X86,
            accelerator={
                Accelerator.NONE: 0.75,
                Accelerator.GPU: 0.25,
                Accelerator.TPU: 0
            },
            cores={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.7,
                Bins.VERY_HIGH: 0.3
            },
            location={
                Location.CLOUD: 0.7,
                Location.MEC: 0.30,
                Location.EDGE: 0,
                Location.MOBILE: 0
            },
            connection={
                Connection.ETHERNET: 1,
                Connection.WIFI: 0,
                Connection.MOBILE: 0
            },
            network={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0.9
            },
            cpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.7,
                Bins.HIGH: 0.25,
                Bins.VERY_HIGH: 0.05
            },
            cpu={
                CpuModel.XEON: 0.7,
                CpuModel.I7: 0.3
            },
            ram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.05,
                Bins.HIGH: 0.45,
                Bins.VERY_HIGH: 0.5
            },
            gpu_vram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.9,
                Bins.VERY_HIGH: 0.1
            },
            gpu_model={
                GpuModel.TURING: 1,
            },
            gpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 1,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 1,
                Disk.SD: 0,
                Disk.NVME: 0,
                Disk.FLASH: 0,
                Disk.HDD: 0
            }
        ),
        Arch.AARCH64: ArchProperties(
            arch=Arch.AARCH64,
            accelerator={
                Accelerator.NONE: 0.2,
                Accelerator.GPU: 0.7,
                Accelerator.TPU: 0.1
            },
            cores={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            location={
                Location.CLOUD: 0,
                Location.MEC: 0.2,
                Location.EDGE: 0.8,
                Location.MOBILE: 0
            },
            connection={
                Connection.ETHERNET: 0.2,
                Connection.WIFI: 0.8,
                Connection.MOBILE: 0
            },
            network={
                Bins.LOW: 0.1,
                Bins.MEDIUM: 0.7,
                Bins.HIGH: 0.2,
                Bins.VERY_HIGH: 0
            },
            cpu_mhz={
                Bins.LOW: 0.1,
                Bins.MEDIUM: 0.8,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            cpu={
                CpuModel.ARM: 1
            },
            ram={
                Bins.LOW: 0.3,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0.2,
                Bins.VERY_HIGH: 0
            },
            gpu_vram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            gpu_model={
                GpuModel.PASCAL: 0.3,
                GpuModel.MAXWELL: 0.4,
                GpuModel.TURING: 0.3
            },
            gpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 0,
                Disk.SD: 0.5,
                Disk.NVME: 0,
                Disk.FLASH: 0.5,
                Disk.HDD: 0
            }
        ),
        Arch.ARM32: ArchProperties(
            arch=Arch.ARM32,
            accelerator={
                Accelerator.NONE: 1,
                Accelerator.GPU: 0,
                Accelerator.TPU: 0
            },
            cores={
                Bins.LOW: 0.5,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            location={
                Location.CLOUD: 0,
                Location.MEC: 0,
                Location.EDGE: 0.9,
                Location.MOBILE: 0.1
            },
            connection={
                Connection.ETHERNET: 0.05,
                Connection.WIFI: 0.85,
                Connection.MOBILE: 0.1
            },
            network={
                Bins.LOW: 0.6,
                Bins.MEDIUM: 0.4,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            cpu_mhz={
                Bins.LOW: 0.5,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            cpu={
                CpuModel.ARM: 1
            },
            ram={
                Bins.LOW: 0.4,
                Bins.MEDIUM: 0.6,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 0,
                Disk.SD: 1,
                Disk.NVME: 0,
                Disk.FLASH: 0,
                Disk.HDD: 0
            },
            gpu_vram={},
            gpu_model={},
            gpu_mhz={},
        )
    }
)

cloud_settings = GeneratorSettings(
    arch={
        Arch.X86: 1,
        Arch.AARCH64: 0,
        Arch.ARM32: 0
    },
    properties={
        Arch.X86: ArchProperties(
            arch=Arch.X86,
            accelerator={
                Accelerator.NONE: 0.75,
                Accelerator.GPU: 0.25,
                Accelerator.TPU: 0
            },
            cores={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.7,
                Bins.VERY_HIGH: 0.3
            },
            location={
                Location.CLOUD: 0.7,
                Location.MEC: 0.30,
                Location.EDGE: 0,
                Location.MOBILE: 0
            },
            connection={
                Connection.ETHERNET: 1,
                Connection.WIFI: 0,
                Connection.MOBILE: 0
            },
            network={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0.9
            },
            cpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.7,
                Bins.HIGH: 0.25,
                Bins.VERY_HIGH: 0.05
            },
            cpu={
                CpuModel.XEON: 0.7,
                CpuModel.I7: 0.3
            },
            ram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.05,
                Bins.HIGH: 0.45,
                Bins.VERY_HIGH: 0.5
            },
            gpu_vram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.9,
                Bins.VERY_HIGH: 0.1
            },
            gpu_model={
                GpuModel.TURING: 1,
            },
            gpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 1,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 1,
                Disk.SD: 0,
                Disk.NVME: 0,
                Disk.FLASH: 0,
                Disk.HDD: 0
            }
        ),
        Arch.AARCH64: ArchProperties(
            arch=Arch.AARCH64,
            accelerator={
                Accelerator.NONE: 0.2,
                Accelerator.GPU: 0.7,
                Accelerator.TPU: 0.1
            },
            cores={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            location={
                Location.CLOUD: 0,
                Location.MEC: 0.2,
                Location.EDGE: 0.8,
                Location.MOBILE: 0
            },
            connection={
                Connection.ETHERNET: 0.2,
                Connection.WIFI: 0.8,
                Connection.MOBILE: 0
            },
            network={
                Bins.LOW: 0.1,
                Bins.MEDIUM: 0.7,
                Bins.HIGH: 0.2,
                Bins.VERY_HIGH: 0
            },
            cpu_mhz={
                Bins.LOW: 0.1,
                Bins.MEDIUM: 0.8,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            cpu={
                CpuModel.ARM: 1
            },
            ram={
                Bins.LOW: 0.3,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0.2,
                Bins.VERY_HIGH: 0
            },
            gpu_vram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            gpu_model={
                GpuModel.PASCAL: 0.3,
                GpuModel.MAXWELL: 0.4,
                GpuModel.TURING: 0.3
            },
            gpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 0,
                Disk.SD: 0.5,
                Disk.NVME: 0,
                Disk.FLASH: 0.5,
                Disk.HDD: 0
            }
        ),
        Arch.ARM32: ArchProperties(
            arch=Arch.ARM32,
            accelerator={
                Accelerator.NONE: 1,
                Accelerator.GPU: 0,
                Accelerator.TPU: 0
            },
            cores={
                Bins.LOW: 0.5,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            location={
                Location.CLOUD: 0,
                Location.MEC: 0,
                Location.EDGE: 0.9,
                Location.MOBILE: 0.1
            },
            connection={
                Connection.ETHERNET: 0.05,
                Connection.WIFI: 0.85,
                Connection.MOBILE: 0.1
            },
            network={
                Bins.LOW: 0.6,
                Bins.MEDIUM: 0.4,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            cpu_mhz={
                Bins.LOW: 0.5,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            cpu={
                CpuModel.ARM: 1
            },
            ram={
                Bins.LOW: 0.4,
                Bins.MEDIUM: 0.6,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 0,
                Disk.SD: 1,
                Disk.NVME: 0,
                Disk.FLASH: 0,
                Disk.HDD: 0
            },
            gpu_vram={},
            gpu_model={},
            gpu_mhz={},
        )
    }
)
