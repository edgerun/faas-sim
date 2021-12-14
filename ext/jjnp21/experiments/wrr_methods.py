"""
Basic idea for these experiments:
    Have a list of "upstreams" with attached fixed response times in a uniform random range
    Have a global hit-list
    Set up LRT/WRR and start sending requests
    Look how long it takes until all items on the hit-list are True i.e. how many requests
"""
import abc
import random
from typing import Dict, List
import pandas as pd
import math
import matplotlib.pyplot as plt


class RandomSampler(abc.ABC):
    @abc.abstractmethod
    def next(self) -> float:
        pass


class LognormalSampler(RandomSampler):
    def __init__(self, mean: float, stdev: float):
        self.mean = math.log((mean ** 2) / math.sqrt(mean ** 2 + stdev ** 2))
        self.variance = math.log(1 + ((stdev ** 2) / (mean ** 2)))

    def next(self) -> float:
        return math.exp(self.mean + self.variance * random.random())


def generate_sampler() -> LognormalSampler:
    mean = random.randint(5, 75)
    stdev = random.randint(int(mean / 2), mean * 2)
    return LognormalSampler(mean, stdev)


class Upstream:
    def __init__(self, response_time_sampler: RandomSampler):
        self.sampler = response_time_sampler

    def simulate_response(self) -> float:
        return self.sampler.next()


class WRR(abc.ABC):
    @abc.abstractmethod
    def init(self, upstreams: Dict[Upstream, float]):
        """
        Initializes the weighted round robin provider
        @param upstreams: dictionary containing the upstreams and the average response time values that get converted to weights
        @return:
        """
        pass

    @abc.abstractmethod
    def next(self) -> Upstream:
        pass

    @abc.abstractmethod
    def update_weights(self, upstreams: Dict[Upstream, float]):
        """
        Updates the weights of the WRR provider. What exactly happens here depends on the actual implementation
        @param upstreams: dictionary containing the upstreams and the average response time values that get converted to weights
        @return:
        """
        pass


def invert_response_times(upstreams: Dict[Upstream, float]) -> Dict[Upstream, float]:
    """
    Simply inverts the response times by calculating 1 / response_time. Useful since then larger values are "better"
    @param upstreams: Dict of upstreams with average response times as values
    @return: A dict of upstreams with the inverted response times as values
    """
    inv = {}
    for u, rt in upstreams.items():
        inv[u] = 1 / rt
    return inv


def response_times_to_weights(upstreams: Dict[Upstream, float], scaling: float = 1.0) -> Dict[Upstream, int]:
    """
    Helper function that converts a dict with Upstreams and their respective response times to weights [0,10]
    @param upstreams: Dict of upstreams with the average response times as values
    @param scaling: If weights should be scaled. Default is flat scaling with a value of 1.0
    @return: Dict of upstreams with the weights between 0 and 10 as values
    """
    weights = {}
    min_val = min(upstreams.values())
    for u, rt in upstreams.items():
        w = int(round(max(1.0, pow(10 / (rt / min_val), scaling))))
        weights[u] = w
    return weights


def gcd(ns: List[int]) -> int:
    max_gcd = min(ns)
    gcd = 1
    for i in range(max_gcd, 0, -1):
        valid = True
        for n in ns:
            if n % i != 0:
                valid = False
                break
        if valid and i > 1:
            gcd = i
            break
    return gcd

class Logger:
    def __init__(self):
        self.rq_log = []
        self.hit_log = []

    def log_rq(self, rq_count: int, response_time: float):
        self.rq_log.append({'request': rq_count, 'rt': response_time})

    def log_hit(self, rq_count: int, hit_percentage: float):
        self.hit_log.append({'request': rq_count, 'hit': hit_percentage})

    def extract_dfs(self) -> Dict[str, pd.DataFrame]:
        return {
            'requests': pd.DataFrame(self.rq_log),
            'hits': pd.DataFrame(self.hit_log),
        }

class ClassicWRR(WRR):
    def __init__(self):
        self.gcd: int = 1
        self.upstreams: List[Upstream] = []
        self.weights: List[int] = []
        self.cw = 0
        self.n = 0
        self.last = -1
        self.max_weight = 1

    def init(self, upstreams: Dict[Upstream, float]):
        weight_dict = response_times_to_weights(upstreams)
        self.upstreams = list(weight_dict.keys())
        self.weights = list(weight_dict.values())
        self.max_weight = max(self.weights)
        self.n = len(self.upstreams)
        self.cw = 0
        self.gcd = gcd(self.weights)
        self.last = 0

    def next(self) -> Upstream:
        while True:
            self.last = (self.last + 1) % self.n
            if self.last == 0:
                self.cw -= self.gcd
                if self.cw <= 0:
                    self.cw = self.max_weight
            if self.weights[self.last] >= self.cw:
                return self.upstreams[self.last]

    def update_weights(self, upstreams: Dict[Upstream, float]):
        self.init(upstreams)


class NextGenWRR(ClassicWRR):
    def __init__(self):
        super().__init__()

    def update_weights(self, upstreams: Dict[Upstream, float]):
        self.gcd = 1
        weight_dict = response_times_to_weights(upstreams)
        self.upstreams = list(weight_dict.keys())
        self.weights = list(weight_dict.values())
        self.max_weight = max(self.weights)
        self.n = len(self.upstreams)


class RandomWRR(WRR):
    def __init__(self):
        self.weights: Dict[Upstream, int] = {}

    def init(self, upstreams: Dict[Upstream, float]):
        self.weights = invert_response_times(upstreams)

    def next(self) -> Upstream:
        return random.choices(list(self.weights.keys()), list(self.weights.values()))[0]

    def update_weights(self, upstreams: Dict[Upstream, float]):
        self.weights = invert_response_times(upstreams)


class SmoothWRR(WRR):

    def __init__(self):
        self.current_weights: Dict[Upstream, float] = {}
        self.upstreams: Dict[Upstream, float] = {}
        self.weight_sum: float = 0.0

    def init(self, upstreams: Dict[Upstream, float]):
        self.upstreams = invert_response_times(upstreams)
        self.current_weights = self.upstreams.copy()
        for u in self.current_weights.keys():
            self.current_weights[u] = 0
        self.weight_sum = sum(self.upstreams.values())

    def next(self) -> Upstream:
        for u, w in self.upstreams.items():
            self.current_weights[u] += w
        choice = max(self.current_weights, key=self.current_weights.get)
        self.current_weights[choice] -= self.weight_sum
        return choice

    def update_weights(self, upstreams: Dict[Upstream, float]):
        self.upstreams = invert_response_times(upstreams)
        self.weight_sum = sum(self.upstreams.values())


class WRRTest:
    def __init__(self, wrr: WRR, upstream_count: int, rqs: int, update_interval: int = 15, lrt_window: float = 30,
                 seed: int = 42):
        self.wrr = wrr
        self.upstream_count = upstream_count
        self.rqs = rqs
        self.window = lrt_window
        self.update_interval = update_interval
        self.upstreams: Dict[Upstream, float] = {}
        self.request_count: Dict[Upstream, int] = {}
        self.last_request: Dict[Upstream, float] = {}
        self.hits: Dict[Upstream, bool] = {}
        self.remaining_hits = self.upstream_count
        self.rq_count = 0
        self.now = 0
        self.logger = Logger()
        random.seed(seed)

    def init(self):
        self.remaining_hits = self.upstream_count
        for _ in range(self.upstream_count):
            upstream = Upstream(generate_sampler())
            self.upstreams[upstream] = 25
            self.hits[upstream] = False
            self.request_count[upstream] = 0
            self.last_request[upstream] = -1
        self.wrr.init(self.upstreams)
        self.rq_count = 0
        self.now = 0
        self.all_hit_rquest_count = 0
        self.response_time_sum = 0

    def run(self):
        self.init()
        while self.rq_count < self.rqs * 500:
            self.now += 1 / self.rqs
            selected_upstream = self.wrr.next()
            rt = selected_upstream.simulate_response()
            self.update_rt_value(selected_upstream, rt)
            self.rq_count += 1
            self.request_count[selected_upstream] += 1
            if not self.hits[selected_upstream]:
                self.hits[selected_upstream] = True
                self.remaining_hits -= 1
                if self.remaining_hits == 0:
                    self.all_hit_rquest_count = self.rq_count
            if self.rq_count % (self.rqs * self.update_interval) == 0:
                self.wrr.update_weights(self.upstreams)
            self.logger.log_rq(self.rq_count, rt)
            self.logger.log_hit(self.rq_count, 1 - (self.remaining_hits / self.upstream_count))

    def update_rt_value(self, upstream: Upstream, last_response_time: float):
        if self.last_request[upstream] == -1:
            self.last_request[upstream] = self.now
            self.upstreams[upstream] = last_response_time
            return
        time_delta = self.now - self.last_request[upstream]
        alpha = 1.0 - math.exp(-time_delta / self.window)
        next_avg_rt = (alpha * last_response_time) + ((1.0 - alpha) * self.upstreams[upstream])
        self.upstreams[upstream] = next_avg_rt
        self.response_time_sum += last_response_time


t1 = WRRTest(RandomWRR(), 500, 5, 15)
t2 = WRRTest(ClassicWRR(), 500, 5, 15)
t3 = WRRTest(SmoothWRR(), 500, 5, 1)
t4 = WRRTest(NextGenWRR(), 500, 5, 15)
t1.run()
t2.run()
t3.run()
t4.run()

results = [('Random', t1), ('Classic', t2), ('Adapted Classic', t4), ('Smooth', t3)]

def name_to_filename(name):
    if name == 'Random':
        return 'random'
    elif name == 'Classic':
        return 'classic'
    elif name == 'Adapted Classic':
        return 'adapted_classic'
    elif name == 'Smooth':
        return 'smooth'

output_path = '/home/jp/Documents/tmp/implementation_experiments/'
for name, test in results:
    path = output_path + name_to_filename(name) + '_hits.csv'
    dfs = test.logger.extract_dfs()
    df = dfs['hits']
    plt.plot(df['request'], df['hit'])
    df.to_csv(path)
plt.legend(['Random', 'Classic', 'Adapted Classic', 'Smooth'])
plt.title('Time until full node coverage')
plt.xlabel('Request #')
plt.ylabel('share of nodes that received at least 1 request')
plt.savefig(output_path + 'hits.png', dpi=1200)
plt.clf()

# plt.figure(dpi=1200)
for name, test in results:
    path = output_path + name_to_filename(name) + '_requests.csv'
    dfs = test.logger.extract_dfs()
    df = dfs['requests'].rolling(200).mean()
    plt.plot(df.index, df['rt'])
    dfs['requests'].to_csv(path)
plt.title('Response Time Convergence')
plt.xlabel('Request #')
plt.ylabel('Rolling response time average')


plt.legend(['Random', 'Classic', 'Adapted Classic', 'Smooth'])

plt.savefig(output_path + 'convergence.png', dpi=1200)
