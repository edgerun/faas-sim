from enum import Enum
from typing import NamedTuple, Dict


class EventType(Enum):
    POD_QUEUED = "pod_queued",
    POD_RECEIVED = "pod_received",
    POD_SCHEDULED = "pod_scheduled"

    # The order is actually also alphabetically, so just implement lt as < on the name
    def __lt__(self, other):
        return self.name < other.name


class LoggingRow(NamedTuple):
    timestamp: float  # When did it happen?
    event: EventType  # What happened?
    value: str  # Which pod was affected?
    additional_attributes: Dict = {}  # What else could be interesting?
