"""
Random generator utilities and wrappers.
"""
import math
import random
import uuid
from typing import Callable
from typing import Generator

import numpy as np
import scipy.stats as st


def buffered_sampler(dist: st.rv_continuous, buffer_size=1000):
    while True:
        sample = dist.rvs(buffer_size)
        yield from sample


def exp_sampler(lambd) -> Generator[float, float, None]:
    return buffered_sampler(st.expon(lambd))


def is_iterable(obj):
    """
    Checks whether the given object is an iterable by calling iter.
    :param obj: the object to check
    :return: true if the object is iterable
    """
    try:
        _ = iter(obj)
    except TypeError:
        return False
    else:
        return True


def randint(a, b=None):
    """
    Returns a random integer in the range of [0,a] or [a,b] depending on whether b is given.
    """
    if b:
        return random.randint(a, b)
    else:
        return random.randint(0, a)


def randfloat(a, b=None):
    """
    Returns a random float in the range of [0,a] or [a,b] depending on whether b is given.
    """
    if b:
        return random.uniform(a, b)
    else:
        return random.uniform(0, a)


def coin_flip(k=None):
    if k:
        return choose([True, False], k=k)
    else:
        return randint(1) == 0


def random_id(prefix, length=8):
    return '%s-%s' % (prefix, str(uuid.uuid4())[:length])


def choose(population, weights=None, k=None):
    """
    Chooses k times from the given population with an optional weighted probability.

    :param population: the population to chose from
    :param weights: the weights attached to each population element
    :param k: the amount of times to chose
    :return: an element of the list if k = None, or a k-sized list of choices
    """
    choice = random.choices(population, weights=weights, k=k or 1)
    return choice if k else choice[0]


def choose_index(population, weights=None, k=None):
    """
    Like choose but returns indices instead of elements.

    :param population: the population to chose from
    :param weights: the weights attached to each population element
    :param k: the amount of times to chose
    :return: an element index of the list if k = None, or a k-sized list of choices
    """
    if type(population) == list:
        n = len(population)
    else:
        n = int(population)

    if weights:
        return choose(range(n), weights=weights, k=k)

    if k:
        return [random.randrange(n) for _ in range(k)]

    return random.randrange(n)


def random_walk_norm(mu_init=0, sigma=0.5):
    """
    Creates an unbounded generator for an MCMC-type random walk using a normal distribution. Toy function.

    :param mu_init: the initial mu value
    :param sigma: the sigma value of the distribution
    :return: a sample
    """
    mu = mu_init

    while True:
        v = random.normalvariate(mu, sigma)
        yield v
        mu = v


def logistic(A, K, B, v, Q, M) -> Callable[[float], float]:
    """
    Creates a generalized logistic function. https://en.wikipedia.org/wiki/Generalised_logistic_function

    :param A: the lower asymptote
    :param K: the upper asymptote
    :param B: the growth rate
    :param v: near which asymptote the growth occurs
    :param Q: Y(0)
    :param M: starting point x_0
    :return: a function
    """

    def f(t):
        return A + (K - A) / ((1 + Q * math.exp(-B * (t - M))) ** (1 / v))

    return f


class RandomSampler:

    def sample(self, size=None):
        """
        Returns a random sample of a population.

        :param size: the size of the return vector. if size is None, a single value is returned.
        :return: scalar or array
        """
        raise NotImplementedError()


class ConstantSampler(RandomSampler):
    """
    Convenience class to replace a random sampler with a constant value.
    """

    def __init__(self, c) -> None:
        super().__init__()
        self.c = c

    def sample(self, size=None):
        if size is None:
            return self.c

        return [self.c] * size


class IntegerSampler(RandomSampler):

    def __init__(self, a, b) -> None:
        super().__init__()
        self.a, self.b = a, b

    def sample(self, size=None):
        if size:
            return [randint(self.a, self.b) for _ in range(size)]
        else:
            return randint(self.a, self.b)


class BoundRejectionSampler(RandomSampler):
    """
    Samples values from another sampler but rejects all values not within the configured bounds.
    """

    def __init__(self, sampler: RandomSampler, minval=None, maxval=None) -> None:
        super().__init__()
        self.sampler = sampler
        self.minval = minval
        self.maxval = maxval

    def sample(self, size=None):
        n = size or 1

        sample = list()

        i = 1.2
        while len(sample) < n:
            # TODO: add additional break condition to defend against very long loops
            count = int(np.ceil(n * i))
            more = self._get_more(count)
            sample.extend(more)

        return sample[0] if not size else np.array(sample[:n])

    def _get_more(self, count):
        more = np.array(self.sampler.sample(count))

        if more.ndim == 1:
            return self._filter_dim1(more)
        elif more.ndim == 2:
            return self._filter_dim2(more)
        else:
            raise NotImplementedError('Cannot filter high dimensional arrays')

    def _filter_dim1(self, more):
        if self.minval is not None:
            more = more[(more >= self.minval)]
        if self.maxval is not None:
            more = more[(more <= self.maxval)]
        return more

    def _filter_dim2(self, more):
        x = more
        if self.minval:
            mx = np.ma.masked_greater_equal(x, self.minval)
            x = x[np.all(mx.mask, axis=1)]
        if self.maxval:
            mx = np.ma.masked_less_equal(x, self.maxval)
            x = x[np.all(mx.mask, axis=1)]
        return x


class BufferedSampler(RandomSampler):
    """
    Wraps another RandomSampler and samples batches from it. This dramatically speeds up code where calls to sample()
    with small sample sizes are frequent, and sampling itself is time consuming (e.g., from a gaussian mixture).
    """

    def __init__(self, sampler, chunk_size=1000) -> None:
        super().__init__()
        self.sampler = sampler
        self.chunk_size = chunk_size
        self.cache = list()

    def sample(self, size=None):
        n = size or 1

        if n > self.chunk_size:
            return self.sampler.sample(n)  # serve large requests directly

        while len(self.cache) < n:
            self.cache.extend(self.sampler.sample(self.chunk_size))

        l, self.cache = self.cache[:n], self.cache[n:]
        return l if size else l[0]

    @classmethod
    def of(cls, sampler: RandomSampler):
        return sampler if isinstance(sampler, BufferedSampler) else BufferedSampler(sampler)


class PopulationSampler(RandomSampler):

    def __init__(self, population, weights=None) -> None:
        super().__init__()

        if isinstance(population, dict) and weights is None:
            self.population = list(population.keys())
            self.weights = list(population.values())
        else:
            self.population = population
            self.weights = weights

    def sample(self, size=None, indices=False):
        fn = choose_index if indices else choose
        return fn(self.population, weights=self.weights, k=size)


class ParameterizedDistribution(RandomSampler):
    """
    Wrapper around scipy's statistical distribution functions. The object holds a distribution type (e.g.
    scipy.stat.norm) and instance parameters (i.e., distribution parameters, scale and location).
    """

    def __init__(self, dist: st.rv_continuous, args, loc=None, scale=None):
        super().__init__()
        self.dist = dist
        self.args = args if is_iterable(args) else [args]
        self.loc = loc
        self.scale = scale

    def sample(self, size=None):
        """
        Draw a random sample from this distribution.

        :param size: [optional] size
        rvs : ndarray or scalar
        """
        kwargs = dict()
        if self.loc is not None:
            kwargs['loc'] = self.loc
        if self.scale is not None:
            kwargs['scale'] = self.scale
        if size is not None:
            kwargs['size'] = size

        return self.dist.rvs(*self.args, **kwargs)

    @property
    def params(self):
        return self.args, self.loc, self.scale

    @property
    def name(self):
        return self.dist.name

    @classmethod
    def fit(cls, dist, x, *args, **kwargs):
        """
        Creates a ParameterizedDistribution for the given distribution by estimating the distribution parameters from
        the given input data.

        :param dist: the scipy distribution function (e.g. scipy.stat.gamma)
        :param x: the example data to fit parameters on to
        :return: a ParameterizedDistribution
        """
        params = dist.fit(x, *args, **kwargs)

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        return cls.create(dist, (arg, loc, scale))

    def plot(self, sample=True):
        import matplotlib.pyplot as plt
        import numpy as np

        x = self.sample(size=5000)
        x_pdf = np.linspace(min(x), max(x), 5000)

        y_pdf = self.dist.pdf(x_pdf, *self.args, **self.kwargs)

        plt.hist(x, bins=int(np.sqrt(len(x))), density=True)
        plt.plot(x_pdf, y_pdf, label='pdf', color='orange')
        plt.title('%s (%s,%s)' % (self.dist.name, self.args, self.kwargs))
        plt.show()

    def pdf(self, x):
        return self.dist.pdf(x, *self.args, **self.kwargs)

    def mean(self):
        return self.dist.mean(*self.args, **self.kwargs)

    def stats(self, moments='mv'):
        kwargs = self.kwargs
        kwargs['moments'] = moments

        return self.dist.stats(*self.args, **kwargs)

    @property
    def kwargs(self):
        kwargs = dict()

        if self.loc is not None:
            kwargs['loc'] = self.loc

        if self.scale is not None:
            kwargs['scale'] = self.scale

        return kwargs

    @classmethod
    def create(cls, dist, params) -> 'ParameterizedDistribution':
        """
        Creates a new ParameterizedDistribution for the given distribution and the given parameter tuple.
        :param dist: the scipy distribution
        :param params: the parameter tuple (*args, loc, scale)
        :return: a ParameterizedDistribution
        """
        arg, loc, scale = params[0], params[1], params[2]
        return ParameterizedDistribution(dist, arg, loc=loc, scale=scale)

    @classmethod
    def uniform(cls, params):
        return cls.create(st.uniform, params)

    @classmethod
    def normal(cls, params):
        return cls.create(st.norm, params)

    @classmethod
    def gamma(cls, params):
        return cls.create(st.gamma, params)

    @classmethod
    def pareto(cls, params):
        return cls.create(st.pareto, params)

    @classmethod
    def exponweib(cls, params):
        return cls.create(st.exponweib, params)

    @classmethod
    def weibull(cls, params):
        return cls.create(st.weibull)

    @classmethod
    def weibull_min(cls, params):
        return cls.create(st.weibull_min, params)

    @classmethod
    def lognorm(cls, params):
        return cls.create(st.lognorm, params)

    @classmethod
    def beta(cls, params):
        return cls.create(st.beta, params)

    def __str__(self) -> str:
        return '[%s %s]' % (self.dist.name, self.params)
