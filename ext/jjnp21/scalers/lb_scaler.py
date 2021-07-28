import abc

from sim.core import Environment


class LoadBalancerScaler(abc.ABC):

    @abc.abstractmethod
    def run(self): ...

    @abc.abstractmethod
    def stop(self): ...
