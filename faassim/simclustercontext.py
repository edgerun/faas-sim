from typing import List, Dict

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

    def get_next_storage_node(self, node: Node) -> str:
        # TODO maybe switch the storage node to different ones than the master?
        return 'ara-clustercloud1'

    def get_init_image_states(self) -> Dict[str, ImageState]:
        # TODO maybe synth other images?
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-1-pre/tags/0.33/
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-2-train/tags/0.33/
        # https://cloud.docker.com/v2/repositories/alexrashed/ml-wf-3-serve/tags/0.33/
        return {
            'alexrashed/ml-wf-1-pre:0.33': ImageState(size={
                'arm': 461473086,
                'arm64': 538015840,
                'amd64': 530300745
            }),
            'alexrashed/ml-wf-2-train:0.33': ImageState(size={
                'arm': 506029298,
                'arm64': 582828211,
                'amd64': 547365470
            }),
            'alexrashed/ml-wf-3-serve:0.33': ImageState(size={
                'arm': 506769993,
                'arm64': 585625232,
                'amd64': 585928717
            })
        }

    def get_bandwidth_graph(self) -> Dict[str, Dict[str, float]]:
        # TODO synthesize the graph -> N^2 entries!
        #  for all nodes in self.list_nodes()
        # 1.25e+6 Byte/s = 10 MBit/s
        # 1.25e+7 Byte/s = 100 MBit/s
        # 1.25e9 Byte/s = 10 GBit/s - assumed for local access
        # The registry is always connected with 100 MBit/s (replicated in both networks)
        # The edge nodes are interconnected with 100 MBit/s
        # The cloud is connected to the edge nodes with 10 MBit/s
        return {
            'ara-clustercloud1': {
                'ara-clustercloud1': 1.25e+9,
                'ara-clustertegra1': 1.25e+6,
                'ara-clusterpi1': 1.25e+6,
                'ara-clusterpi2': 1.25e+6,
                'ara-clusterpi3': 1.25e+6,
                'ara-clusterpi4': 1.25e+6,
                'registry': 1.25e+7
            },
            'ara-clustertegra1': {
                'ara-clustercloud1': 1.25e+6,
                'ara-clustertegra1': 1.25e+9,
                'ara-clusterpi1': 1.25e+7,
                'ara-clusterpi2': 1.25e+7,
                'ara-clusterpi3': 1.25e+7,
                'ara-clusterpi4': 1.25e+7,
                'registry': 1.25e+7
            },
            'ara-clusterpi1': {
                'ara-clustercloud1': 1.25e+6,
                'ara-clustertegra1': 1.25e+7,
                'ara-clusterpi1': 1.25e+9,
                'ara-clusterpi2': 1.25e+7,
                'ara-clusterpi3': 1.25e+7,
                'ara-clusterpi4': 1.25e+7,
                'registry': 1.25e+7
            },
            'ara-clusterpi2': {
                'ara-clustercloud1': 1.25e+6,
                'ara-clustertegra1': 1.25e+7,
                'ara-clusterpi1': 1.25e+7,
                'ara-clusterpi2': 1.25e+9,
                'ara-clusterpi3': 1.25e+7,
                'ara-clusterpi4': 1.25e+7,
                'registry': 1.25e+7
            },
            'ara-clusterpi3': {
                'ara-clustercloud1': 1.25e+6,
                'ara-clustertegra1': 1.25e+7,
                'ara-clusterpi1': 1.25e+7,
                'ara-clusterpi2': 1.25e+7,
                'ara-clusterpi3': 1.25e+9,
                'ara-clusterpi4': 1.25e+7,
                'registry': 1.25e+7
            },
            'ara-clusterpi4': {
                'ara-clustercloud1': 1.25e+6,
                'ara-clustertegra1': 1.25e+7,
                'ara-clusterpi1': 1.25e+7,
                'ara-clusterpi2': 1.25e+7,
                'ara-clusterpi3': 1.25e+7,
                'ara-clusterpi4': 1.25e+9,
                'registry': 1.25e+7
            }
        }
