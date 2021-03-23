from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Pod, Node
from skippy.core.predicates import Predicate

from ext.raith21.model import Accelerator
from sim.oracle.oracle import ResourceOracle, FetOracle


class HasEnoughRamPredicate(Predicate):
    def __init__(self, resource_oracle: ResourceOracle):
        self.resource_oracle = resource_oracle

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        """ram is in percentage, relative to node total ram"""
        host = node.name
        image = pod.spec.containers[0].image
        needed_ram = self.resource_oracle.get_resources(host, image)['ram']
        ram_in_use = 0
        for running_pod in node.pods:
            running_image = running_pod.spec.containers[0].image
            ram_in_use += self.resource_oracle.get_resources(host, running_image)['ram']
        return ram_in_use + needed_ram <= 1


class CanRunPred(Predicate):
    def __init__(self, fet_oracle: FetOracle, resource_oracle: ResourceOracle):
        self.fet_oracle = fet_oracle
        self.resource_oracle = resource_oracle

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        host = node.name[:node.name.rindex('_')] if '_' in node.name else node.name
        if host == 'registry':
            return False
        image = pod.spec.containers[0].image
        return self.fet_oracle.sample(host, image) is not None \
            and self.resource_oracle.get_resources(host, image) is not None


class NodeHasAcceleratorPred(Predicate):
    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        accelerator_label = 'device.edgerun.io/accelerator'
        if accelerator_label in pod.spec.labels.keys():
            return pod.spec.labels[accelerator_label] == node.labels.get(accelerator_label, '')
        else:
            return True


class NodeHasFreeTpu(Predicate):
    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        accelerator_label = 'device.edgerun.io/accelerator'
        if pod.spec.labels.get(accelerator_label, '') == str(Accelerator.TPU.name):
            for running_pod in node.pods:
                if running_pod.spec.labels.get(accelerator_label, '') == 'TPU':
                    return False

            return True
        else:
            return True


class NodeHasFreeGpu(Predicate):

    def __init__(self, use_vram: bool = False):
        self.use_vram = use_vram

    def passes_predicate(self, context: ClusterContext, pod: Pod, node: Node) -> bool:
        accelerator_label = 'device.edgerun.io/accelerator'
        vram_label = 'device.edgerun.io/vram'
        if pod.spec.labels.get(accelerator_label, '') == str(Accelerator.GPU.name):
            if not self.use_vram:
                for running_pod in node.pods:
                    if running_pod.spec.labels.get(accelerator_label, '') == str(Accelerator.GPU.name):
                        return False
                return True

            vram_needed = int(pod.spec.labels[vram_label])
            vram_size = int(node.labels[vram_label])
            reserved_vram = 0
            for running_pod in node.pods:
                if running_pod.spec.labels.get(accelerator_label, '') == str(Accelerator.GPU.name):
                    reserved_vram += int(running_pod.spec.labels.get(vram_label, '0'))

            return vram_needed + reserved_vram <= vram_size
        else:
            return True
