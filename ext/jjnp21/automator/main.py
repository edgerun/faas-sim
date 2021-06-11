import os
from typing import List

from ext.jjnp21.automator.execution import run_experiment
from ext.jjnp21.automator.experiment import Experiment, Result
from multiprocessing import Queue, Process, JoinableQueue


class ExperimentWorker(Process):
    def __init__(self, task_queue: JoinableQueue, result_queue: Queue):
        Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            result = next_task()
            self.task_queue.task_done()
            self.result_queue.put(result)


class ExperimentTask:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def __call__(self, *args, **kwargs) -> Result:
        return run_experiment(self.experiment)


class ExperimentRunAutomator:
    experiments: List[Experiment]
    worker_count: int
    iterations: int
    task_queue: JoinableQueue
    result_queue: Queue

    def __init__(self, experiments: List[Experiment], worker_count: int = 4, iterations: int = 1):
        # TODO implement iterations
        self.experiments = experiments
        self.worker_count = worker_count
        self.iterations = iterations

    def run(self, reporting_interval: int = 15) -> List[Result]:
        thread_count = max(os.cpu_count(), self.worker_count)
        self.task_queue = JoinableQueue()
        self.result_queue = Queue()
        workers = [ExperimentWorker(self.task_queue, self.result_queue) for i in range(0, thread_count)]
        for w in workers:
            w.start()
        for e in self.experiments:
            self.task_queue.put(ExperimentTask(e))
        # Put in one poison pill signifier to make workers stop once everything is done
        for _ in range(0, thread_count):
            self.task_queue.put(None)
        self.task_queue.join()
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get(timeout=5))
        return results
