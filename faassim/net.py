import logging
from typing import List, Dict

import simpy

from core.model import Node

logger = logging.getLogger(__name__)


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

    def run(self):
        env = self.env
        size = self.size
        route = self.route
        source = route.source
        hops = route.hops
        sink = route.destination

        # find the link that has the lowest available bandwidth
        bottleneck = min([link.get_max_allocatable(self) for link in hops])
        # allocate that bandwidth in all links
        goodput = min([link.allocate(self, bottleneck) for link in hops])
        # calculate the simulation time
        bytes_remaining = self.size
        transmission_time = bytes_remaining / goodput  # remaining seconds

        timer = env.now

        connection_time = ((route.rtt * 1.5) / 1000)  # rough estimate of TCP connection establish time
        while connection_time > 0:
            started = env.now
            try:
                yield env.timeout(connection_time)
                break
            except simpy.Interrupt as interrupt:
                connection_time = connection_time - (env.now - started)

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
                logger.debug('%-5.2f sending %s -[%d]-> {%s} interrupted: %s (sent: %d, remaining: %d)',
                             env.now, source.name, size, sink.name, interrupt.cause, self.sent, bytes_remaining)

                bottleneck = min([link.get_max_allocatable(self) for link in hops])
                goodput = min([link.allocate(self, bottleneck) for link in hops])
                transmission_time = bytes_remaining / goodput  # set new time remaining

        logger.debug('%-5.2f sending %s -[%d]-> {%s} completed in %.2fs',
                     env.now, source.name, size, sink.name, env.now - timer)

        for link in hops:
            link.free(self)

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

    allocation: Dict[Flow, int]
    goodput_per_flow: int

    tags: dict

    def __init__(self, bandwidth: int = 100, tags=None) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.allocation = dict()
        self.tags = tags or dict()

    def get_max_allocatable(self, flow: Flow):
        flows = len(self.allocation)

        if flow not in self.allocation:
            flows += 1  # +1 if the flow is new

        # fair_per_flow is the maximum bandwidth a flow can get if there are no other flows that require less
        fair_per_flow = int(self.bandwidth / flows)

        # flows that require less than the fair value may keep it
        reserved = {k: v for k, v in self.allocation.items() if v < fair_per_flow}
        allocatable = self.bandwidth - sum(reserved.values())

        # these are the flows competing for the remaining bandwidth
        competing_flows = flows - len(reserved)
        if competing_flows:
            allocatable_per_flow = int(allocatable / competing_flows)
        else:
            allocatable_per_flow = allocatable

        return max(fair_per_flow, allocatable_per_flow)

    def reallocate(self):
        # same principle as get_max_allocatable()
        flows = len(self.allocation)

        if flows == 0:
            return

        fair_per_flow = int(self.bandwidth / flows)

        reserved = {k: v for k, v in self.allocation.items() if v < fair_per_flow}
        allocatable = self.bandwidth - sum(reserved.values())

        competing_flows = [flow for flow in self.allocation.keys() if flow not in reserved]

        if competing_flows:
            allocatable_per_flow = int(allocatable / len(competing_flows))

            for flow in competing_flows:
                self.allocation[flow] = allocatable_per_flow

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

    def allocate(self, flow: Flow, requested_bandwidth):
        if self.allocation.get(flow, 0) == requested_bandwidth:
            return self.get_goodput_bps(flow)

        self.allocation[flow] = requested_bandwidth
        self.reallocate()

        for f in self.allocation.keys():
            if f is flow:
                continue
            process = f.process
            if process and process.is_alive:
                process.interrupt('update available bandwidth per flow %d bytes/sec' % self.get_goodput_bps(f))

        return self.get_goodput_bps(flow)

    def free(self, flow: Flow):
        del self.allocation[flow]
        self.reallocate()

        for f in self.allocation.keys():
            process = f.process
            if process and process.is_alive:
                process.interrupt('update available bandwidth per flow %d bytes/sec' % self.get_goodput_bps(f))
