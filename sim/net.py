import logging

from ether.core import Flow

logger = logging.getLogger(__name__)


class LowBandwidthException(BaseException):
    pass


def SafeFlow(*args, bw_threshold=0.1, **kwargs):
    """
    Creates a new ether.core.Flow but throws a LowBandwidthException if the allocatable bandwidth in the flow is less
    than the given threshold. This is used to terminate simulations that produce infeasible function placements.

    :param args: the arguments for the ether.core.Flow
    :param bw_threshold: the minimal bandwidth threshold (default 0.1)
    :param kwargs: the kwargs for ether.core.Flow
    :return: a flow
    """
    flow = Flow(*args, **kwargs)
    try:
        bottleneck = min(flow.route.hops, key=lambda l: l.max_allocatable)
    except:
        print('lol')
    if bottleneck.max_allocatable <= bw_threshold:
        logger.error('potential for flow %s: %.4f', flow.route, bottleneck.max_allocatable)
        raise LowBandwidthException()

    return flow
