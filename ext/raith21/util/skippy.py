from skippy.core.priorities import DataLocalityPriority, LatencyAwareImageLocalityPriority, BalancedResourcePriority, \
    LocalityTypePriority, CapabilityPriority

from ext.raith21.util import predicates


def get_predicates(fet_oracle, resource_oracle):
    return predicates.get_predicates(fet_oracle, resource_oracle)


def get_priorities(balance_weight: float = 1, latency_weight: float = 1, locality_weight: float = 1,
                   data_weight: float = 1, cap_weight: float = 1):
    return [
        (balance_weight, BalancedResourcePriority()),
        (latency_weight, DataLocalityPriority()),
        (data_weight, LatencyAwareImageLocalityPriority()),
        (locality_weight, LocalityTypePriority()),
        (cap_weight, CapabilityPriority())
    ]
