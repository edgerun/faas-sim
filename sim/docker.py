from collections import defaultdict
from typing import List, Tuple, NamedTuple

from sim.core import Node, Environment
from sim.net import SafeFlow


class ImageProperties(NamedTuple):
    name: str
    size: int
    tag: str = 'latest'
    arch: str = None


class Registry:

    def __init__(self) -> None:
        super().__init__()
        # images['edgerun/go-telemd']['latest'] = [ImageProperties]
        self.images = defaultdict(lambda: defaultdict(list))

    def put(self, image: ImageProperties):
        self.images[image.name][image.tag].append(image)

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
    started = env.now
    # TODO: there's a lot of potential to improve fidelity here: consider image layers, simulate extraction time, etc.

    # find the image in the registry with the node's architecture
    images = env.registry.find(image_str, arch=node.arch)
    if not images:
        raise ValueError('image not in registry: %s arch=%s' % (image_str, node.arch))
    image = images[0]

    if image in node.docker_images:
        yield env.timeout(0)

    size = image.size

    if size <= 0:
        yield env.timeout(0)

    # # FIXME: crude simulation of layer sharing (90% across images is shared)
    # num_images = len(env.cluster.images_on_nodes[node.name]) - 1
    # if num_images > 0:
    #     size = size * 0.1

    route = env.topology.get_route(env.topology.get_registry(), node)
    flow = SafeFlow(env, size, route)

    yield flow.start()

    for hop in route.hops:
        env.metrics.log_network(size, 'docker_pull', hop)
    env.metrics.log_flow(size, env.now - started, route.source, route.destination, 'docker_pull')
