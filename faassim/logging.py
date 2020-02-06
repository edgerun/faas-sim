from datetime import datetime, timedelta
from typing import Dict, NamedTuple


class Clock:
    def now(self) -> datetime:
        raise NotImplementedError()


class WallClock(Clock):

    def now(self) -> datetime:
        return datetime.now()


class Record(NamedTuple):
    measurement: str
    time: int
    fields: Dict
    tags: Dict


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


class RuntimeLogger:
    def __init__(self, clock=None) -> None:
        self.records = list()
        self.clock = clock or WallClock()

    def get(self, name, **tags):
        return lambda x: self.log(name, x, None, **tags)

    def log(self, metric, value, time=None, **tags):
        """
        Call l.log('cpu_load', .65, host='server0', region='us-west') or

        :param metric: the name of the measurement
        :param value: the measurement value
        :param time: the (optional) time, otherwise now will be used
        :param tags: additional tags describing the measurement
        :return:
        """
        if time is None:
            time = self._now()

        if type(value) == dict:
            fields = value
        else:
            fields = {
                'value': value
            }

        self._store_record(Record(metric, time, fields, tags))

    def _store_record(self, record: Record):
        self.records.append(record)

    def _now(self):
        return self.clock.now()


class NullLogger(RuntimeLogger):
    """
    Null logger does nothing.
    """

    def log(self, name, value, time=None, **tags):
        pass


class PrintLogger(RuntimeLogger):

    def _store_record(self, record: Record):
        super()._store_record(record)
        print('[log]', record)
