from dataclasses import dataclass, field
from typing import Dict

from ext.raith21.model import Location, Disk, Bins, Accelerator, Arch, Connection, CpuModel, GpuModel


@dataclass
class ArchProperties:
    arch: Arch
    accelerator: Dict[Accelerator, float]
    cores: Dict[Bins, float]
    disk: Dict[Disk, float]
    location: Dict[Location, float]
    connection: Dict[Connection, float]
    network: Dict[Bins, float]
    cpu_mhz: Dict[Bins, float]
    cpu: Dict[CpuModel, float]
    ram: Dict[Bins, float]
    gpu_vram: Dict[Bins, float] = field(default_factory=dict)
    gpu_mhz: Dict[Bins, float] = field(default_factory=dict)
    gpu_model: Dict[GpuModel, float] = field(default_factory=dict)

    @property
    def values(self):
        return [
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


@dataclass
class Device:
    id: str
    arch: Arch
    accelerator: Accelerator
    cores: Bins
    disk: Disk
    location: Location
    connection: Connection
    network: Bins
    cpu_mhz: Bins
    cpu: CpuModel
    ram: Bins

    @property
    def labels(self) -> Dict[str, str]:
        return {
            'device.edgerun.io/arch': str(self.arch.name),
            'device.edgerun.io/accelerator': str(self.accelerator.name),
            'device.edgerun.io/cores': str(self.cores.name),
            'device.edgerun.io/location': str(self.location.name),
            'device.edgerun.io/connection': str(self.connection.name),
            'device.edgerun.io/network': str(self.network.name),
            'device.edgerun.io/cpu_mhz': str(self.cpu_mhz.name),
            'device.edgerun.io/cpu': str(self.cpu.name),
            'device.edgerun.io/ram': str(self.ram.name),
            'device.edgerun.io/disk': str(self.disk.name)
        }

    def copy(self):
        return Device(
            id=self.id,
            arch=self.arch,
            accelerator=self.accelerator,
            cores=self.cores,
            disk=self.disk,
            location=self.location,
            connection=self.connection,
            network=self.network,
            cpu_mhz=self.cpu_mhz,
            cpu=self.cpu,
            ram=self.ram
        )


@dataclass
class GpuDevice(Device):
    vram: Bins
    gpu_mhz: Bins
    gpu_model: GpuModel

    @property
    def labels(self) -> Dict[str, str]:
        super_labels = super().labels
        super_labels['device.edgerun.io/vram_bin'] = str(self.vram.name)
        super_labels['device.edgerun.io/gpu_mhz'] = str(self.gpu_mhz.name)
        super_labels['device.edgerun.io/gpu_model'] = str(self.gpu_model.name)
        return super_labels

    def copy(self):
        return GpuDevice(
            id=self.id,
            arch=self.arch,
            accelerator=self.accelerator,
            cores=self.cores,
            disk=self.disk,
            location=self.location,
            connection=self.connection,
            network=self.network,
            cpu_mhz=self.cpu_mhz,
            cpu=self.cpu,
            ram=self.ram,
            vram=self.vram,
            gpu_mhz=self.gpu_mhz,
            gpu_model=self.gpu_model
        )
