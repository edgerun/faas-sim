import abc

from skippy.core.model import SchedulingResult, Pod


class LoadBalancerScheduler(abc.ABC):
    def schedule(self, pod: Pod) -> SchedulingResult: ...
