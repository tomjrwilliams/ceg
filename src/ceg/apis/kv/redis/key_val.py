from __future__ import annotations
from dataclasses import dataclass

from typing import (
    Type,
    TypeVar,
    ClassVar,
    final,
)
from dataclasses import dataclass, replace

from redis import Redis

from ..core import (
    KeyVal,
    KeyValConfig,
    ExampleKeyVal,
    # _,
    KeyO,
    ValO,
    ValT,
)

from ... import logs

logger, log = logs.logger(
    __file__, streams=dict(formatter=True),
)


from functools import lru_cache
@lru_cache(maxsize=4)
def redis_conn(db: str, cache: bool = False):

    conn = Redis(
        db,
        port=7777,
        decode_responses=False,
        single_connection_client=True,
        socket_timeout=10.,
    )
    conn_get = conn.get

    @lru_cache(maxsize=16)
    def get(key):
        return conn_get(key)
    
    if cache:
        conn.get = get
    return conn


@dataclass(frozen=True)
class RedisKeyValConfig(KeyValConfig):
    db: str
    name: str
    conn: Redis | None = None
    # if plural, dict, then overload on string literals when subclass (?)
    # table: Redis.Table | None = None
    # spec: dict | None = None


ID = TypeVar("ID", bound=int)
KeyI = TypeVar("KeyI", bound=int | str)
ValI = TypeVar(
    "ValI", bound=bytes
)


@dataclass(frozen=True)
class RedisKeyVal(KeyVal[ID, KeyI, ValI, KeyO, ValO]):
    """KeyVal[ID, KeyI, ValI, KeyO, ValO]"""

    config: RedisKeyValConfig

    # TODO: we can in theory set this different ot generic param in the subclass def - is there a way to err that?

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    chunked: int | None = None  # indicates chunk size

    CONFIG: ClassVar[Type[RedisKeyValConfig]] = (
        RedisKeyValConfig
    )

    def prefix_key(self, key: KeyI):
        return f"{self.config.name}/{key}"
    def prepare(self):
        config = self.config
        return replace(self, config=replace(
            config,
            conn=redis_conn(
                self.config.db,
                # cache="Value" in self.config.name
                # decode_responses=False
            )
        ))

    def identify(
        self,
        key: KeyI,
    ) -> None | int:

        conn = self.config.conn
        assert conn is not None, self

        i = conn.exists(self.prefix_key(key))
        if i == 0:
            return None
        return int(i)

    def lookup(self, id: int) -> KeyI:
        raise ValueError(id)

    def select(
        self,
        key: KeyI,
    ) -> None | ValI:
        conn = self.config.conn
        assert conn is not None, self

        key = self.prefix_key(key)
        log("INFO")(redis_get=key)
        val = conn.get(key)

        if val is None:
            return val
        
        assert isinstance(val, self.type_val_i), val

        return val

    def iselect(
        self,
        key: KeyI,
    ) -> tuple[ID | None, ValI | None]:
        conn = self.config.conn
        assert conn is not None, self

        val = conn.get(self.prefix_key(key))

        if val is None:
            return None, None
        
        assert isinstance(val, self.type_val_i), val

        return 1, val

    def insert(
        self,
        key: KeyI,
        val: ValI,
    ) -> ID:

        conn = self.config.conn
        assert conn is not None, self.config

        res = conn.set(self.prefix_key(key), val)

        return 1

    def update(
        self,
        key: KeyI,
        val: ValI,
        i: int | None = None,
    ) -> ID:
        conn = self.config.conn
        assert conn is not None, self.config
        return self.insert(key, val)


# @final
# @dataclass(frozen=True)
# class ExampleRedisKeyVal(
#     ExampleKeyVal,
#     RedisKeyVal[int, str, bytes, str, float]
# ):
#     """
#     >>> j, k, l, updated = ("j", "k", "l", [])
#     >>> kv = ExampleRedisKeyVal(
#     ...     on_update = lambda k, v: updated.append((k, v))
#     ... ).prepare()
#     >>> kv.get(k)
#     >>> v = kv.set(k, 0.0)
#     >>> assert v == 0.0 == kv.get(k), kv.get(k)
#     >>> assert (1, 0.0) == kv.iget(k), kv.iget(k)
#     >>> v = kv.set(j, 1.0)
#     >>> v = kv.set(k, 2.0)
#     >>> assert (2, 1.0) == kv.iget(j)
#     >>> assert (1, 2.0) == kv.iget(k)
#     >>> assert updated == [(k, 2.0)]
#     """

#     config: RedisKeyValConfig = RedisKeyValConfig(
#         db="localhost", name="ExampleKeyVal"
#     )


__all__ = [
    "RedisKeyValConfig",
    "ID",
    "KeyI",
    "ValI",
    "RedisKeyVal",
]
