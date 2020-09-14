from collections import defaultdict
from typing import List, Tuple, NamedTuple, Dict

from sim.core import Node, Environment
from sim.net import SafeFlow
from sim.topology import DockerRegistry


class ImageProperties(NamedTuple):
    name: str
    size: int
    tag: str = 'latest'
    arch: str = None


class Registry:
    """
    The registry keeps track of container images and their properties.
    """

    # images['edgerun/go-telemd']['latest'] = [ImageProperties]
    images: Dict[str, Dict[str, List[ImageProperties]]]

    def __init__(self) -> None:
        super().__init__()
        self.images = defaultdict(lambda: defaultdict(list))

    def put(self, image: ImageProperties):
        self.images[image.name][image.tag].append(image)

    def put_all(self, images: List[ImageProperties]):
        for image in images:
            self.put(image)

    def find(self, image: str, arch=None) -> List[ImageProperties]:
        repository, tag = split_image_name(image)

        images = self.images[repository][tag]

        if arch:
            images = [image for image in images if image.arch == arch or image.arch is None]

        return images


def split_image_name(image: str) -> Tuple[str, str]:
    parts = image.split(':', maxsplit=1)

    if len(parts) == 1:
        return parts[0], 'latest'

    return parts[0], parts[1]


def pull(env: Environment, image_str: str, node: Node):
    """
    Simulate a docker pull command of the given image on the given node.

    :param env: the simulation environment
    :param image_str: the name of the image (<repository[:tag]>)
    :param node: the node on which to run the pull command
    :return: a simpy process (a generator)
    """
    started = env.now
    # TODO: there's a lot of potential to improve fidelity here: consider image layers, simulate extraction time, etc.
    #  e.g., docker pull on a 13MB container takes about 5 seconds. the simulated time at 120 MBit/sec would be <1s

    # find the image in the registry with the node's architecture
    images = env.registry.find(image_str, arch=node.arch)
    if not images:
        raise ValueError('image not in registry: %s arch=%s' % (image_str, node.arch))
    image = images[0]

    node_state = env.get_node_state(node.name)
    if node_state:
        if image in node_state.docker_images:
            return
        else:
            node_state.docker_images.add(image)

    size = image.size

    if size <= 0:
        return

    # # FIXME: crude simulation of layer sharing (90% across images is shared)
    # num_images = len(env.cluster.images_on_nodes[node.name]) - 1
    # if num_images > 0:
    #     size = size * 0.1

    route = env.topology.route(DockerRegistry, node)
    flow = SafeFlow(env, size, route)

    yield flow.start()

    # for hop in route.hops:
    #     env.metrics.log_network(size, 'docker_pull', hop)
    env.metrics.log_flow(size, env.now - started, route.source, route.destination, 'docker_pull')
