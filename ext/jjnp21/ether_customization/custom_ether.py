from ether.core import Flow
import simpy
import logging

logger = logging.getLogger(__name__)


def remove_without_rebalance(flow: Flow):
    for link in flow.route.hops:
        link.num_flows -= 1
        del link.allocation[flow]
        link.recalculate_max_allocatable()


def add_without_rebalance(flow: Flow):
    allocated_bandwidth = min([link.max_allocatable for link in flow.route.hops])

    for link in flow.route.hops:
        link.num_flows += 1
        link.recalculate_max_allocatable()
        link.allocation[flow] = allocated_bandwidth


class UninterruptingFlow(Flow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        if connection_time > 0:
            yield env.timeout(connection_time)

        add_without_rebalance(self)
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
            remove_without_rebalance(self)
