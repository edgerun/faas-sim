import abc
from typing import Optional, TypeVar, Generic

from faas.util.rwlock import ReadWriteLock

I = TypeVar('I')


# TODO maybe this can be put into the faas project
class KeyValueStorage(abc.ABC, Generic[I]):

    def get(self, key: str, default=None) -> Optional[I]: ...

    def put(self, key: str, value: I): ...


class InMemoryKeyValueStorage(KeyValueStorage[I]):

    def __init__(self):
        self.data = {}
        self.rw_lock = ReadWriteLock()

    def get(self, key: str, default=None) -> Optional[I]:
        with self.rw_lock.lock.gen_rlock():
            return self.data.get(key, default)

    def put(self, key: str, value: I):
        with self.rw_lock.lock.gen_wlock():
            self.data[key] = value
