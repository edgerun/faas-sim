from skippy.core.model import Pod, SchedulingResult

from ext.jjnp21.schedulers.lb_schedulers.lb_scheduler import LoadBalancerScheduler


"""
This should basically
"""

class OsmoticLoadBalancerScheduler(LoadBalancerScheduler):
    def __init__(self):
        pass

    def schedule(self, pod: Pod) -> SchedulingResult:
        return super().schedule(pod)