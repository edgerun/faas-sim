from ext.raith21 import predicates
from ext.raith21.priorities import CapabilityMatchingPriority, ContentionPriority, ExecutionTimePriority


def get_predicates(fet_oracle, resource_oracle):
    return predicates.get_predicates(fet_oracle, resource_oracle)


def get_priorities(fet_oracle, resource_oracle, capability_weight: float = 1, contention_weight: float = 1,
                   fet_weight: float = 1):
    return [
        (capability_weight, CapabilityMatchingPriority()),
        (contention_weight, ContentionPriority(fet_oracle, resource_oracle)),
        (fet_weight, ExecutionTimePriority(fet_oracle))
    ]
