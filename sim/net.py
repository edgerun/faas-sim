import logging

from ether.core import Flow

logger = logging.getLogger(__name__)


class LowBandwidthException(BaseException):
    pass


def SafeFlow(*args, bw_threshold=0.1, **kwargs):
    flow = Flow(*args, **kwargs)

    bottleneck = min(flow.route.hops, key=lambda l: l.max_allocatable)

    if bottleneck.max_allocatable <= bw_threshold:
        logger.error('potential for flow %s: %.4f', flow.route, bottleneck.max_allocatable)
        raise LowBandwidthException()

    return flow
