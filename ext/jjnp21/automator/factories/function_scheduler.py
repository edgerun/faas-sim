from ext.jjnp21.schedulers.random_scheduler import RandomScheduler
# from ext.raith21.main import resource_oracle, fet_oracle
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.oracles import Raith21FetOracle, Raith21ResourceOracle
from ext.raith21.predicates import CanRunPred, NodeHasAcceleratorPred, NodeHasFreeGpu, NodeHasFreeTpu
from ext.raith21.resources import ai_resources_per_node_image
from ext.raith21.util import vanilla
from sim.core import Environment
from skippy.core.scheduler import Scheduler


class FunctionSchedulerFactory:
    constructor_kwargs = {}

    def set_constructor_args(self, **kwargs):
        self.constructor_kwargs = kwargs

    def create(self, env: Environment):
        raise Exception('Do not use this class directly. Use an actual implementation.')


class DefaultFunctionSchedulerFactory(FunctionSchedulerFactory):
    def create(self, env: Environment):
        fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
        resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
        predicates = []
        predicates.extend(Scheduler.default_predicates)
        predicates.extend([
            CanRunPred(fet_oracle, resource_oracle),
            NodeHasAcceleratorPred(),
            NodeHasFreeGpu(),
            NodeHasFreeTpu()
        ])
        priorities = vanilla.get_priorities()
        sched_params = {
            'percentage_of_nodes_to_score': 100,
            'priorities': priorities,
            'predicates': predicates
        }
        scheduler = Scheduler(env.cluster, **sched_params)
        return scheduler


class RandomFunctionSchedulerFactory(FunctionSchedulerFactory):
    def create(self, env: Environment):
        fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
        resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
        predicates = []
        predicates.extend(Scheduler.default_predicates)
        predicates.extend([
            CanRunPred(fet_oracle, resource_oracle),
            NodeHasAcceleratorPred(),
            NodeHasFreeGpu(),
            NodeHasFreeTpu()
        ])
        return RandomScheduler.create(env, predicates)
