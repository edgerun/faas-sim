from typing import Generator

import scipy.stats as st


def buffered_sampler(dist: st.rv_continuous, buffer_size=1000):
    while True:
        sample = dist.rvs(buffer_size)
        yield from sample


def exp_sampler(lambd) -> Generator[float, float, None]:
    return buffered_sampler(st.expon(lambd))