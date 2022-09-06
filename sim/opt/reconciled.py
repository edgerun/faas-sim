from faas.opt.api import Optimizer
from faas.opt.reconciled import ReconciliationOptimizationDaemon

from sim.core import Environment


class SimReconciliationOptimizationDaemon(ReconciliationOptimizationDaemon):

    def __init__(self, reconcile_interval: float, env: Environment, optimizer: Optimizer):
        super().__init__(optimizer)
        self.env = env
        self.reconcile_interval = reconcile_interval

    def sleep(self):
        yield self.env.timeout(self.reconcile_interval)

    def run(self):
        self.optimizer.run()
        while self.is_running:
            yield from self.sleep()
            yield from self.optimizer.run()


