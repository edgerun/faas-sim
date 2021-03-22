import ast
from typing import Dict

from skippy.core.clustercontext import ClusterContext
from skippy.core.model import Pod, Node
from skippy.core.priorities import Priority, _scale_scores

from sim.oracle.oracle import FetOracle, ResourceOracle


class CapabilityMatchingPriority(Priority):
    def map_node_score(self, context: ClusterContext, pod: Pod, node: Node) -> int:
        priority = 0
        raw_requirements = pod.spec.labels.get('device.edgerun.io/requirements', None)
        if raw_requirements is None:
            return 0

        node_caps = dict(filter(lambda label: 'device.edgerun.io' in label[0], node.labels.items()))

        requirements: Dict[str, Dict[str, float]] = ast.literal_eval(raw_requirements)
        for capability in node_caps.items():
            label = requirements.get(capability[0], None)
            if label is not None:
                priority += label.get(capability[1], 0)

        return priority

    def reduce_mapped_score(self, context: ClusterContext, pod: Pod, nodes: [Node], node_scores: [int]) -> [int]:
        return _scale_scores(node_scores, context.max_priority)


class ExecutionTimePriority(Priority):

    def __init__(self, fet_oracle: FetOracle):
        super().__init__()
        self.fet_oracle = fet_oracle

    def map_node_score(self, context: ClusterContext, pod: Pod, node: Node) -> int:
        fet = self.fet_oracle.sample(node.name, pod.spec.containers[0].image)
        return -fet if fet is not None else 0

    def reduce_mapped_score(self, context: ClusterContext, pod: Pod, nodes: [Node], node_scores: [int]) -> [int]:
        return _scale_scores(node_scores, context.max_priority)


class ContentionPriority(Priority):

    def __init__(self, fet_oracle: FetOracle, resource_oracle: ResourceOracle):
        super().__init__()
        self.fet_oracle = fet_oracle
        self.resource_oracle = resource_oracle

    def _get_disk_speed(self, disk: str):
        if 'NVME' in disk:
            return 2500e6
        if 'SSD' in disk:
            return 500e6
        if 'HDD' in disk:
            return 250e6
        if 'FLASH' in disk:
            return 150e6
        if 'SD' in disk:
            return 50e6

        return 1

    def _get_net_speed(self, location: str):
        # TODO change this if deployment has more than two types of networks
        if 'CLOUD' in location:
            return 1000e6
        else:
            return 125e6

    def map_node_score(self, context: ClusterContext, pod: Pod, node: Node) -> int:
        image = pod.spec.containers[0].image
        host = node.name[:node.name.rindex('_')]
        usage = self.resource_oracle.get_resources(host, image)
        # normalize per second
        pod_blkio = usage['blkio']
        pod_net = usage['net']
        pod_cpu = usage['cpu']
        pod_gpu = usage['gpu']

        # calculate the current average of all running pods on this node
        running_cpu = 0
        running_blkio = 0
        running_net = 0
        running_gpu = 0
        counter = 0
        for running_pod in node.pods:
            image = running_pod.spec.containers[0].image
            usage = self.resource_oracle.get_resources(host, image)
            if usage != None:
                running_blkio += usage['blkio']
                running_net += usage['net']
                running_cpu += usage['cpu']
                running_gpu += usage['gpu']
                counter += 1
        if counter > 0:
            running_net /= self._get_net_speed(node.labels.get('device.edgerun.io/location'))
            running_blkio /= self._get_disk_speed(node.labels.get('device.edgerun.io/disk'))

        pod_net /= self._get_net_speed(node.labels.get('device.edgerun.io/location'))
        pod_blkio /= self._get_disk_speed(node.labels.get('device.edgerun.io/disk'))

        # multiply to make range larger -> may help when converting to int
        pod_usage = (pod_blkio + pod_net + pod_cpu + pod_gpu)
        running_usage = (running_blkio + running_net + running_gpu + running_cpu)
        priority = pod_usage - running_usage
        return priority

    def reduce_mapped_score(self, context: ClusterContext, pod: Pod, nodes: [Node], node_scores: [int]) -> [int]:
        return _scale_scores(node_scores, context.max_priority)
