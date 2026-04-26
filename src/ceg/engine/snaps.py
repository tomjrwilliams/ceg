from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    overload,
    cast,
    final,
)
import functools
from pydantic import (
    TypeAdapter,
    BaseModel,
    Field,
)

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

from ..apis import kv, t, assert_is
from .events import *

import os

TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")


@dataclass(frozen=True)
class SnapParser:
    snap: Snap

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)


@functools.lru_cache(maxsize=128, typed=True)
def snap_parser(cls: Type[Snap]) -> Type[SnapParser]:
    return dataclass(frozen=True)(
        type(
            cls.__name__ + "Parser",
            (SnapParser,),
            {
                "snap": Field(discriminator="type"),
                "__annotations__": {"snap": cls},
            },
        )
    )


@dataclass(frozen=True)
class Snap:
    """
    >>> @dataclass(frozen=True)
    ... class EventEx(Event):
    ...     i: int | None = Event.i_field()
    ...     type: Literal['Ex'] = Event.type_field("Ex")
    >>> @dataclass(frozen=True)
    ... class SnapA(Snap):
    ...     a: int
    ...     type: Literal['A'] = Snap.type_field("A")
    >>> @dataclass(frozen=True)
    ... class SnapB(Snap):
    ...     b: float
    ...     type: Literal['B'] = Snap.type_field("B")
    >>> @dataclass(frozen=True)
    ... class SnapParserAB(SnapParser):
    ...     snap: SnapA | SnapB = Snap.union_field()
    >>> a, b = SnapA(EventEx("a"), "a", 1), SnapB(EventEx("b"), "b", 1.)
    >>> _ = SnapParserAB(a) # init parser type
    >>> Snap.parse_json(a.dump_bytes(), SnapParserAB)
    SnapA(event=Event(obj='a'), obj='a', a=1)
    >>> Snap.parse_json(b.dump_bytes(), SnapParserAB)
    SnapB(event=Event(obj='b'), obj='b', b=1.0)
    """

    obj: str  # NOTE: object path
    handled: t.frozen.dict[str, int] = t.frozen.dict()
    updates: Snap | None = None

    buffer: int | None = None
    triage: bool = False
    latest: t.frozen.dict[str, float] = t.frozen.dict()

    @property
    def updates_path(self):
        return f"{self.obj}-Updates"

    def set_handled(self, id: str, i: tuple[int, float]):
        i = assert_is(i[0], int)
        curr = self.handled.get(id, -1)
        assert i == curr + 1, (self, curr, i)
        handled = self.handled.set(id, i)
        self = replace(self, handled=handled)
        if not self.buffer:
            return self
        elif self.updates is None:
            return replace(
                self,
                updates=EMPTY_UPDATES.set_handled(id, i),
            )
        else:
            return replace(
                self,
                updates=self.updates.set_handled(id, i),
            )

    def set_latest(self, id: str, latest: tuple[int, float] | None):
        # NOTE: assume always called after handled
        # so if buffer, we should always have self.updates

        if latest is None:
            if not self.buffer:
                return replace(self, latest=frozendict())
            return replace(self, updates=self.updates.set_latest(latest))

        t = assert_is(latest[1], float)
        curr = self.latest.get(id, 0.)
        assert curr < t, (self, curr, t)
        latest = self.latest.set(id, t)
        self = replace(self, latest=latest)
        
        if not self.buffer:
            return self
        
        return replace(self, updates=self.updates.set_latest(latest))

    def clear_handled(self):
        return replace(self, handled=t.frozen.dict())

    def merge_updates(self, updates: Snap):
        return replace(
            handled=self.handled | updates.handled,
            updates=updates,
            latest=self.latest | updates.latest,
        )

    @classmethod
    def new(
        cls,
        obj: str,
        handled: t.frozen.dict[str, tuple[int, float]] = t.frozen.dict(),
        **kwargs,
    ):
        return cls(obj, handled, **kwargs)

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[SnapParser]:
        return snap_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[SnapParser],
    ) -> Snap:
        parser: SnapParser = into.ta().validate_json(data)
        return parser.snap

    @classmethod
    def get(
        cls,
        key: str,
        kv: SnapKeyVal,
    ) -> Snap | None:
        self = kv.get(key)
        updates = kv.get(f"{key}-Updates")
        if updates is None:
            return self
        return self.merge_updates(updates)

    def set(self, kv: SnapKeyVal) -> Snap:
        if not self.buffer:
            return kv.set(self.obj, self)
        elif self.updates is None:
            return kv.set(self.obj, self)
        elif len(self.updates.handled) < self.buffer:
            kv.set(self.updates_path, self.updates)
            return self
        _ = kv.set(
            self.updates_path, self.updates.clear_handled()
        )
        return kv.set(self.obj, replace(self, updates=None))


# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class SnapKeyVal(kv.KeyValStrKeyIOBytesValI[int, Snap]):
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes] = bytes
    type_key_o: Type[str] = str
    type_val_o: Type[Snap] = Snap

    parser: Type[SnapParser] | None = None

    def encode_val_type(self, t: Type[Snap]) -> Type[bytes]:
        return bytes

    def encode_val(self, k: str, v: Event) -> bytes:
        return v.dump_bytes()

    def decode_val(self, k: str, v: bytes) -> Snap:
        assert self.parser is not None, self
        return Snap.parse_json(v, into=self.parser)


@dataclass(frozen=True)
class SnapUpdates(Snap):
    type: str = Snap.type_field("SnapUpdates")


EMPTY_UPDATES = SnapUpdates("")


@dataclass(frozen=True)
class SnapKeyValSQLite(
    SnapKeyVal, kv.SQLiteKeyVal[int, str, bytes, str, Snap]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> @dataclass(frozen=True)
    ... class EventEx(Event):
    ...     i: int | None = Event.i_field()
    ...     type: Literal['Ex'] = Event.type_field("Ex")
    >>> @dataclass(frozen=True)
    ... class SnapEx(Snap):
    ...     event: EventEx
    ...     type: Literal['A'] = Snap.type_field("A")
    >>> kv = SnapKeyValSQLite.new(
    ...     TEST_DB,
    ...     SnapEx.parser(),
    ...     on_update = lambda k, v: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k)
    >>> v = kv.set(k, SnapEx(EventEx(k), k))
    >>> assert v == SnapEx(EventEx(k), k) == kv.get(k), kv.get(k)
    >>> assert (1, SnapEx(EventEx(k), k)) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, SnapEx(EventEx(j), j))
    >>> v = kv.set(k, SnapEx(EventEx(l), l))
    >>> assert (2, SnapEx(EventEx(j), j)) == kv.iget(j)
    >>> assert (1, SnapEx(EventEx(l), l)) == kv.iget(k)
    >>> assert updated == [(k, SnapEx(EventEx(l), l))]
    """

    @classmethod
    def new(
        cls,
        db: str,
        parser: Type[SnapParser],
        table: str = "Snaps",
        on_update: Callable[[str, Snap], None]
        | None = None,
    ) -> SnapKeyValSQLite:
        self = cls(
            config=kv.SQLiteKeyValConfig(db=db, name=table),
            on_update=on_update,
            parser=parser,
        )
        return cast(SnapKeyValSQLite, self.prepare())

@dataclass(frozen=True)
class SnapKeyValRedis(
    SnapKeyVal, kv.RedisKeyVal[int, str, bytes, str, Snap]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> @dataclass(frozen=True)
    ... class EventEx(Event):
    ...     i: int | None = Event.i_field()
    ...     type: Literal['Ex'] = Event.type_field("Ex")
    >>> @dataclass(frozen=True)
    ... class SnapEx(Snap):
    ...     event: EventEx
    ...     type: Literal['A'] = Snap.type_field("A")
    >>> kv = SnapKeyValSQLite.new(
    ...     TEST_DB,
    ...     SnapEx.parser(),
    ...     on_update = lambda k, v: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k)
    >>> v = kv.set(k, SnapEx(EventEx(k), k))
    >>> assert v == SnapEx(EventEx(k), k) == kv.get(k), kv.get(k)
    >>> assert (1, SnapEx(EventEx(k), k)) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, SnapEx(EventEx(j), j))
    >>> v = kv.set(k, SnapEx(EventEx(l), l))
    >>> assert (2, SnapEx(EventEx(j), j)) == kv.iget(j)
    >>> assert (1, SnapEx(EventEx(l), l)) == kv.iget(k)
    >>> assert updated == [(k, SnapEx(EventEx(l), l))]
    """

    @classmethod
    def new(
        cls,
        db: str,
        parser: Type[SnapParser],
        table: str = "Snaps",
        on_update: Callable[[str, Snap], None]
        | None = None,
    ) -> SnapKeyValRedis:
        self = cls(
            config=kv.RedisKeyValConfig(db=db, name=table),
            on_update=on_update,
            parser=parser,
        )
        return cast(SnapKeyValRedis, self.prepare())



__all__ = [
    "SnapParser",
    "Snap",
    "SnapKeyVal",
    "SnapKeyValSQLite",
    "SnapKeyValRedis",
]
