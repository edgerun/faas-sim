from typing import List

from core.clustercontext import ClusterContext
from core.model import Node, ImageState, Capacity


class SimulationClusterContext(ClusterContext):

    def __init__(self):
        super().__init__()

        # TODO synthesize lots of nodes based on our test-environment
        self.nodes = [
            Node(name='ara-clustercloud1',
                 capacity=Capacity(cpu_millis = 4 * 1000,
                                   memory= 8 * 1024 * 1024 * 1024),
                 allocatable=Capacity(cpu_millis = 4 * 1000,
                                   memory= 8 * 1024 * 1024 * 1024),
                 labels={
                     'beta.kubernetes.io/arch': 'amd64'
                 }),
            Node(name='ara-clustertegra1',
                 capacity=Capacity(cpu_millis = 4 * 1000,
                                   memory= 8 * 1024 * 1024 * 1024),
                 allocatable=Capacity(cpu_millis = 4 * 1000,
                                   memory= 8 * 1024 * 1024 * 1024),
                 labels={
                     'beta.kubernetes.io/arch': 'arm64',
                     'capability.skippy.io': 'nvidia-cuda-10'
                 })
        ] + [
            Node(name='ara-clusterpi{}'.format(i),
                 capacity=Capacity(cpu_millis = 4 * 1000,
                                   memory= 1 * 1024 * 1024 * 1024),
                 allocatable=Capacity(cpu_millis = 4 * 1000,
                                   memory= 1 * 1024 * 1024 * 1024),
                 labels={
                     'beta.kubernetes.io/arch': 'arm'
                 })
             for i in range(1,5)
        ]

    def list_nodes(self) -> List[Node]:
        return self.nodes
