from datetime import datetime, timedelta

from faas.system import Clock


class SimulatedClock(Clock):
    """
    This Clock maps simulation time to a real datetime by interpreting simulation time of the environment as seconds,
    and adding them to a specified start date.
    """

    def __init__(self, env, start: datetime = None) -> None:
        super().__init__()
        self.env = env
        self.start = start or datetime.now()

    def now(self):
        return self.from_simtime(self.env.now)

    def from_simtime(self, seconds) -> datetime:
        return self.start + timedelta(seconds=seconds)
