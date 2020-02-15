import logging
import time
from collections import deque, defaultdict
from typing import List, Dict, NamedTuple, Tuple

import simpy

from core.clustercontext import BandwidthGraph
from core.model import Node
from sim.simclustercontext import SimulationClusterContext

logger = logging.getLogger(__name__)

Internet = 'internet'
Registry = Node('registry')


class Route:
    source: Node
    destination: Node
    hops: List['Link']
    rtt: float = 0  # round-trip latency in milliseconds

    def __init__(self, source: Node, destination: Node, hops: List['Link'], rtt=0) -> None:
        super().__init__()
        self.source = source
        self.destination = destination
        self.hops = hops
        self.rtt = rtt

    def __str__(self) -> str:
        return f'Route[{self.source} ->{self.hops}-> {self.destination}]'


class Flow:
    sent: int
    size: int
    route: Route

    process: simpy.Process

    def __init__(self, env: simpy.Environment, size: int, route: Route) -> None:
        super().__init__()
        self.env = env
        self.size = size  # size in bytes
        self.route = route
        self.sent = 0

    def start(self):
        self.process = self.env.process(self.run())
        return self.process

    def get_goodput_bps(self):
        return min([link.get_goodput_bps(self) for link in self.route.hops])

    def run(self):
        env = self.env
        size = self.size
        route = self.route
        source = route.source
        hops = route.hops
        sink = route.destination

        if not hops:
            raise ValueError('no hops in route from %s to %s' % (source, sink))

        timer = env.now
        connection_time = ((route.rtt * 1.5) / 1000)  # rough estimate of TCP connection establish time
        while connection_time > 0:
            yield env.timeout(connection_time)

        add_and_rebalance(self)
        goodput = self.get_goodput_bps()

        if goodput <= 0:
            raise ValueError
        # calculate the simulation time
        bytes_remaining = self.size
        transmission_time = bytes_remaining / goodput  # remaining seconds

        try:
            while True:
                started = env.now

                try:
                    logger.debug('%-5.2f sending %s -[%d]-> {%s} at %d bytes/sec',
                                 env.now, source.name, size, sink.name, goodput)
                    yield env.timeout(transmission_time)
                    break
                except simpy.Interrupt as interrupt:
                    self.sent += goodput * (env.now - started)
                    if self.sent >= size:
                        break  # was interrupted, but actually sent everything already

                    bytes_remaining = size - self.sent
                    logger.debug('%-5.2f sending %s -[%d]-> {%s} interrupted, new bw = %.2f (sent: %d, remaining: %d)',
                                 env.now, source.name, size, sink.name, interrupt.cause, self.sent, bytes_remaining)

                    goodput = self.get_goodput_bps()
                    if goodput <= 0:
                        raise ValueError
                    transmission_time = bytes_remaining / goodput  # set new time remaining

            logger.debug('%-5.2f sending %s -[%d]-> {%s} completed in %.2fs',
                         env.now, source.name, size, sink.name, env.now - timer)
        finally:
            remove_and_rebalance(self)

    def establish(self):
        env = self.env
        route = self.route

        connection_time = ((route.rtt * 1.5) / 1000)  # rough estimate of TCP connection establish time
        while connection_time > 0:
            started = env.now
            try:
                yield env.timeout(connection_time)
                break
            except simpy.Interrupt as interrupt:
                connection_time = connection_time - (env.now - started)


class Link:
    bandwidth: int  # MBit/s
    tags: dict

    # calculated by rebalance
    allocation: Dict[Flow, float]
    num_flows: int
    max_allocatable: float

    def __init__(self, bandwidth: int = 100, tags=None) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.tags = tags or dict()

        self.allocation = dict()
        self.num_flows = 0
        self.max_allocatable = bandwidth

    def recalculate_max_allocatable(self):
        num_flows = self.num_flows
        bandwidth = self.bandwidth

        if num_flows == 0:
            self.max_allocatable = bandwidth
            return

        # fair_per_flow is the maximum bandwidth a flow can get if there are no other flows that require less
        fair_per_flow = bandwidth / num_flows

        # flows that require less than the fair value may keep it
        reserved = {k: v for k, v in self.allocation.items() if v < fair_per_flow}
        allocatable = bandwidth - sum(reserved.values())

        # these are the flows competing for the remaining bandwidth
        competing_flows = num_flows - len(reserved)
        if competing_flows:
            allocatable_per_flow = allocatable / competing_flows
        else:
            allocatable_per_flow = allocatable

        self.max_allocatable = max(fair_per_flow, allocatable_per_flow)

    def get_goodput_bps(self, flow: Flow):
        """
        Returns the TCP goodput for a flow in bytes per second.
        """
        # TODO: calculate more accurately
        # TODO: use some degradation function? https://pdos.csail.mit.edu/~rtm/papers/icnp97-web.pdf

        if flow not in self.allocation:
            return None

        allocated = self.allocation[flow]
        practical_bw = allocated * 125000
        goodput_magic_number = 0.97  # rough estimate of goodput (~ TCP overhead)

        return practical_bw * goodput_magic_number

    def __str__(self) -> str:
        return f'Link({hex(id(self))}){self.tags}'

    def __repr__(self):
        return self.__str__()


def remove_and_rebalance(flow: Flow):
    # first, collect all affected flows and links
    affected_flows, affected_links = collect_subnet(flow)
    affected_flows.remove(flow)

    for link in flow.route.hops:
        link.num_flows -= 1
        del link.allocation[flow]
        link.recalculate_max_allocatable()

    rebalance(flow, affected_flows, affected_links)


def add_and_rebalance(flow: Flow):
    # first, collect all affected flows and links
    affected_flows, affected_links = collect_subnet(flow)

    for link in flow.route.hops:
        link.num_flows += 1
        link.recalculate_max_allocatable()

    rebalance(flow, affected_flows, affected_links)


def rebalance(triggering_flow, affected_flows, affected_links):
    # holds all allocations that have changed (some may be unaffected)
    allocation: Dict[Flow, float] = dict()

    while affected_flows:
        bottlenecks = {flow: min([link.max_allocatable for link in flow.route.hops]) for flow in affected_flows}
        flow: Flow = min(bottlenecks, key=lambda k: bottlenecks[k])
        request = bottlenecks[flow]

        changed = False

        for link in flow.route.hops:
            if link.allocation.get(flow) == request:
                continue
            changed = True
            link.allocation[flow] = request
            link.recalculate_max_allocatable()

        if changed:
            allocation[flow] = request

        del bottlenecks[flow]
        affected_flows.remove(flow)

    for flow, bw in allocation.items():
        if flow is triggering_flow:
            continue
        if not flow.process.is_alive:
            continue
        flow.process.interrupt(bw)

    # logger.info(' >> new allocation:')
    # for link in affected_links:
    #     logger.info(' - %s (%.2f)', link, link.bandwidth)
    #     for flow, bw in link.allocation.items():
    #         logger.info('   - %8.2f %s', bw, flow.route)

    return allocation


def collect_subnet(flow: Flow):
    # first, collect all affected flows and links
    affected_links = set()
    affected_flows = set()

    stack = set()
    stack.add(flow)

    while stack:
        elem = stack.pop()
        if isinstance(elem, Link):
            if elem in affected_links:
                continue
            affected_links.add(elem)

            flows = elem.allocation.keys()
            stack.update(flows)

        elif isinstance(elem, Flow):
            if elem in affected_flows:
                continue
            affected_flows.add(elem)

            links = elem.route.hops
            stack.update(links)
        else:
            raise ValueError('element of type %s not handled: %s' % (type(elem), elem))

    return affected_flows, affected_links


class Edge(NamedTuple):
    source: object
    target: object
    directed: bool = False


class Graph:
    """
    A graph.

    Example use for an example network topology:

    # a,b,c are three hosts in a LAN, and have up/downlink to the cloud
    n = ['a', 'b', 'c', 'down', 'up', 'cloud']

    edges = [
        Edge(n[0], n[1]),  # a,b
        Edge(n[1], n[2]),  # b,c
        Edge(n[0], n[2]),  # a,c
        Edge(n[3], n[0], True),  # down, a
        Edge(n[3], n[1], True),  # down, b
        Edge(n[3], n[2], True),  # down, c
        Edge(n[0], n[4], True),  # up, a
        Edge(n[1], n[4], True),  # up, b
        Edge(n[2], n[4], True),  # up, c
        Edge(n[4], n[5], True),  # up, cloud
        Edge(n[5], n[3], True),  # cloud, down
    ]

    t = Graph(n, edges)

    print(t.path(n[0], n[1]))  # ['a', 'b']
    print(t.path(n[0], n[5]))  # ['a', 'up', 'cloud']
    print(t.path(n[5], n[0]))  # ['cloud', 'down', 'a']
    """
    nodes: List
    edges: List[Edge]

    def __init__(self, nodes: List, edges: List[Edge]) -> None:
        super().__init__()
        self.nodes = nodes
        self.edges = edges
        self.index = None

    def create_index(self):
        index = defaultdict(list)

        for edge in self.edges:
            n1 = edge.source
            n2 = edge.target
            index[n1].append(n2)
            if not edge.directed:
                index[n2].append(n1)

        self.index = dict(index)

    def successors(self, node):
        return self.index[node] if self.index else self._successors_gen(node)

    def _successors_gen(self, node):
        for edge in self.edges:
            if edge.source == node:
                yield edge.target
            elif not edge.directed and edge.target == node:
                yield edge.source

    def path(self, source, destination) -> List:
        if source == destination:
            return []

        queue = deque([source])
        visited = set()
        parents = dict()

        while queue:
            node = queue.popleft()

            if node == destination:
                # found the destination, collect the path
                path = []
                cur = destination
                while cur:
                    path.append(cur)
                    cur = parents.get(cur)
                return list(reversed(path))

            visited.add(node)

            for successor in self.successors(node):
                if successor is node:
                    continue
                if successor in visited:
                    continue
                if successor not in parents:
                    parents[successor] = node
                queue.append(successor)

        return []


class Topology(Graph):
    internet = Internet
    registry = Registry

    def __init__(self, nodes: List, edges: List[Edge]) -> None:
        super().__init__(nodes, edges)
        self._bandwidth_graph = None
        self._registry = None
        self._route_cache: Dict[Tuple[Node, Node], Route] = dict()

    def get_route(self, source: Node, destination: Node):
        k = (source, destination)

        if k not in self._route_cache:
            path = self.path(source, destination)
            hops = [node for node in path if isinstance(node, Link)]
            self._route_cache[k] = Route(source, destination, hops)

        return self._route_cache[k]

    def get_registry(self):
        if not self._registry:
            self._registry = self.get_host(Registry.name)

        return self._registry

    def get_host(self, name):
        for node in self.nodes:
            if isinstance(node, Node) and node.name == name:
                return node

        return None

    def get_hosts(self):
        result = list()

        for node in self.nodes:
            if not isinstance(node, Node):
                continue

            result.append(node)

        return result

    def get_links(self):
        links = set()

        for edge in self.edges:
            if isinstance(edge.source, Link):
                links.add(edge.source)
            if isinstance(edge.target, Link):
                links.add(edge.target)

        return list(links)

    def get_bandwidth_graph(self) -> BandwidthGraph:
        if self._bandwidth_graph is None:
            self._bandwidth_graph = self.create_bandwidth_graph_parallel()

        return self._bandwidth_graph

    def create_bandwidth_graph(self) -> BandwidthGraph:
        """
        From a topology, create the reduced bandwidth graph required by the ClusterContext.
        :return: bandwidth[from][to] = bandwidth in bytes per second
        """
        then = time.time()

        nodes = self.get_hosts()
        graph = defaultdict(dict)

        # route each node to each other and find the highest available bandwidth
        n = len(nodes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    n1 = nodes[i].name
                    graph[n1][n1] = 1.25e+8  # essentially models disk read from itself as 1GBit/s
                    continue

                n1 = nodes[i]
                n2 = nodes[j]

                route = self.get_route(n1, n2)

                if not route.hops:
                    logger.debug('no route from %s to %s', n1, n2)
                    continue

                bandwidth = min([link.bandwidth for link in route.hops])  # get the maximal available bandwidth
                bandwidth = bandwidth * 125000  # link bandwidth is given in mbit/s: * 125000 = bytes/s

                graph[n1.name][n2.name] = bandwidth

        logger.info('creating bandwidth graph took %.4f seconds', (time.time() - then))

        return graph

    def create_bandwidth_graph_parallel(self, p=4) -> BandwidthGraph:
        import multiprocessing as mp
        """
        From a topology, create the reduced bandwidth graph required by the ClusterContext.
        :return: bandwidth[from][to] = bandwidth in bytes per second
        """
        then = time.time()

        nodes = self.get_hosts()

        # route each node to each other and find the highest available bandwidth
        n = len(nodes)

        parts = partition(list(range(n)), p)
        partitions = [(p, nodes) for p in parts]

        g = dict()

        with mp.Pool(p) as pool:
            part_results = pool.map(self._get_graph_part, partitions)
            for result in part_results:
                g.update(result)

        logger.info('creating bandwidth graph took %.4f seconds', (time.time() - then))

        return g

    def _get_graph_part(self, part):
        irange, nodes = part
        n = len(nodes)

        graph = defaultdict(dict)
        for i in irange:
            for j in range(n):
                if i == j:
                    n1 = nodes[i].name
                    graph[n1][n1] = 1.25e+8  # essentially models disk read from itself as 1GBit/s
                    continue

                n1 = nodes[i]
                n2 = nodes[j]

                route = self.get_route(n1, n2)

                if not route.hops:
                    logger.debug('no route from %s to %s', n1, n2)
                    continue

                bandwidth = min([link.bandwidth for link in route.hops])  # get the maximal available bandwidth
                bandwidth = bandwidth * 125000  # link bandwidth is given in mbit/s: * 125000 = bytes/s

                graph[n1.name][n2.name] = bandwidth

        return graph

    def create_cluster_context(self):
        # remove registry if present
        nodes = [node for node in self.get_hosts() if node.name != Registry.name]

        return SimulationClusterContext(nodes, self.get_bandwidth_graph())


def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]
