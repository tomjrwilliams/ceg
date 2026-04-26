from dataclasses import dataclass

from typing import (
    Type,
    TypeVar,
    ClassVar,
    cast,
    final,
)
from dataclasses import dataclass, replace
import numpy as np
from redis import Redis
from ..core import (
    KeyChunk,
    KeyChunkConfig,
    ExampleKeyChunk,
    # _,
    KeyO,
    ValO,
    ValT,
    #
    HEAD,
    BODY,
    TAIL,
)
from .key_val import ID, KeyI, ValI, RedisKeyValConfig

from functools import lru_cache
@lru_cache(maxsize=4)
def redis_conn(db: str):
    return Redis(
        db,
        port=7777,
        decode_responses=False,
        single_connection_client=True,
        socket_timeout=10.,
    )

@dataclass(frozen=True)
class RedisKeyChunkConfig(
    RedisKeyValConfig,
    KeyChunkConfig,
):
    pass


@dataclass(frozen=True)
class RedisKeyChunk(
    KeyChunk[ID, KeyI, ValI, KeyO, ValO, ValT]
):
    config: RedisKeyChunkConfig

    # TODO: we can in theory set this different ot generic param in the subclass def - is there a way to err that?

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    CONFIG: ClassVar[Type[RedisKeyChunkConfig]] = (
        RedisKeyChunkConfig
    )

    def prepare_key(self, key: KeyI, chunk: int):
        return f"{self.config.name}/{key}/{chunk}"
    
    def prepare(self):
        config = self.config
        return replace(self, config=replace(
            config,
            conn=redis_conn(
                self.config.db,
                # decode_responses=False
            )
        ))

    def identify(
        self,
        key: KeyI,
        chunk: int,
    ) -> None | int:

        conn = self.config.conn
        assert conn is not None, self

        i = conn.exists(self.prepare_key(key, chunk))

        return None if i == 0 else i

    def lookup(
        self, id: int
    ) -> tuple[KeyI, int] | tuple[None, None]:
        raise ValueError(id)

    def select(
        self,
        key: KeyI,
        chunk: int,
    ) -> None | ValI:
        conn = self.config.conn
        assert conn is not None, self
        
        res = conn.get(self.prepare_key(key, chunk))

        return res

    def iselect(
        self,
        key: KeyI,
        chunk: int,
    ) -> tuple[ID | None, ValI | None]:
        conn = self.config.conn
        assert conn is not None, self

        res = self.select(key, chunk)

        if res is None:
            return None, None
        return 1, res

    def insert(
        self,
        key: KeyI,
        val: ValI,
        chunk: int,
    ) -> ID:

        conn = self.config.conn
        assert conn is not None, self.config

        res = conn.set(self.prepare_key(key, chunk), val)

        return 1

    def update(
        self,
        key: KeyI,
        val: ValI,
        chunk: int,
        i: int | None = None,
    ) -> ID:
        return self.insert(key, val, chunk)


@final
@dataclass(frozen=True)
class ExampleRedisKeyChunk(
    ExampleKeyChunk,
    RedisKeyChunk[int, str, bytes, str, np.ndarray, float],
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> v_arr = lambda v: np.array([v])
    >>> kv = ExampleRedisKeyChunk(
    ...     on_update = lambda k, v, c: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k, HEAD)
    >>> v = kv.set(k, v_arr(0.), HEAD)
    >>> assert v == v_arr(0.) == kv.get(k, HEAD)
    >>> assert (1, v_arr(0.)) == kv.iget(k, HEAD), kv.iget(k, HEAD)
    >>> _ = kv.set(j, v_arr(1.), HEAD)
    >>> _ = kv.set(k, v_arr(2.), HEAD)
    >>> assert (2, v_arr(1.)) == kv.iget(j, HEAD)
    >>> assert (1, v_arr(2.)) == kv.iget(k, HEAD)
    >>> assert updated == [(k, v_arr(2.))]
    >>> _ = kv.append_head(k, 1.)
    >>> _ = kv.append_head(k, 0.)
    >>> _ = kv.extend_head(k, np.array([2., 1.]))
    >>> _ = kv.append_tail(j, 2.)
    >>> _ = kv.append_tail(j, 3.)
    >>> _ = kv.extend_tail(j, np.array([2., 1., 0.]))
    >>> _ = kv.append_head(j, 0.)
    >>> kv.get(k, HEAD), kv.get(k, BODY)
    (array([2.]), array([1., 0., 1., 2.]))
    >>> kv.get(j, HEAD), kv.get(j, BODY), kv.get(j, TAIL)
    (array([0.]), array([1., 2., 3., 2., 1.]), array([0.]))
    """

    config: RedisKeyChunkConfig = RedisKeyChunkConfig(
        size=1, db="localhost", name="ExampleKeyChunk"
    )


__all__ = [
    "RedisKeyChunkConfig",
    "RedisKeyChunk",
]
