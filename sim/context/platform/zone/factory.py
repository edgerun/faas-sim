from typing import List

from faas.context import ZoneService
from faas.context.platform.zone.inmemory import InMemoryZoneService


def create_zone_service(zones: List[str]) -> ZoneService:
    return InMemoryZoneService(zones)
