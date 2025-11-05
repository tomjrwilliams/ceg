from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    overload,
    cast,
    final,
)
import datetime as dt
import datetime
import functools
from pydantic import (
    TypeAdapter,
    BaseModel,
    Field,
)

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass
from dataclasses import replace

from frozendict import frozendict

from ..apis import kv

import os
TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

@dataclass(frozen=True)
class EventParser:
    event: Event

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)

@functools.lru_cache(maxsize=128, typed=True)
def event_parser(cls: Type[Event]) -> Type[EventParser]:
    return dataclass(frozen=True)(type(
        cls.__name__ + "Parser",
        (EventParser,),
        {
            "event": Field(discriminator="type"),
            "__annotations__": {"event": cls}
        }
    ))

@dataclass(frozen=True)
class Event:
    """
    >>> @dataclass(frozen=True)
    ... class EventA(Event):
    ...     a: int
    ...     i: int | None = Event.i_field()
    ...     type: Literal['A'] = Event.type_field("A")
    >>> @dataclass(frozen=True)
    ... class EventB(Event):
    ...     b: float
    ...     i: int | None = Event.i_field()
    ...     type: Literal['B'] = Event.type_field("B")
    >>> @dataclass(frozen=True)
    ... class EventParserAB(EventParser):
    ...     event: EventA | EventB = Event.union_field()
    >>> e_a, e_b = EventA("a", 1), EventB("b", 1.)
    >>> _ = EventParserAB(e_a) # init parser type
    >>> Event.parse_json(e_a.dump_bytes(), EventParserAB)
    EventA(obj='a', a=1)
    >>> Event.parse_json(e_b.dump_bytes(), EventParserAB) 
    EventB(obj='b', b=1.0)
    """
    obj: str # NOTE: object path

    @property
    def i(self) -> int | None: ...

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def i_field(cls, repr: bool = False):
        return Field(default=None, repr=repr)

    @classmethod
    def parser(cls) -> Type[EventParser]:
        return event_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[EventParser]
    ) -> Event:
        parser: EventParser = into.ta().validate_json(data)
        return parser.event

    def append(self, q: EventQueue) -> Event:
        i, e, dt = q.iappend(self)
        return replace(self, i = i)

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class EventQueue(
    kv.QueueTimestampAsFloat[int, bytes, Event]
):
    
    type_id: Type[int] = int
    type_dt_i: Type[float] = float
    type_val_i: Type[bytes] = bytes
    type_dt_o: Type[dt.datetime] = dt.datetime
    type_val_o: Type[Event] = Event

    parser: Type[EventParser] | None = None

    def encode_val_type(self, t: Type[Event]) -> Type[bytes]:
        return bytes

    def encode_val(self, v: Event) -> bytes:
        return v.dump_bytes()

    def decode_val(self, v: bytes) -> Event:
        assert self.parser is not None, self
        return Event.parse_json(v, into=self.parser)

    def head_with_i(
        self, 
        n: int = 1,
        where_i: list[tuple[kv.Comp, int]] | None = None,
        where_dt: list[tuple[kv.Comp, dt.datetime]] | None = None,
    ) -> list[Event]:
        i_s, e_s, dt_s = self.ihead(n=n, where_i=where_i, where_dt=where_dt)
        return [
            replace(e, i=i) for e, i in zip(e_s, i_s)
        ]

    def tail_with_i(
        self, 
        n: int = 1,
        where_i: list[tuple[kv.Comp, int]] | None = None,
        where_dt: list[tuple[kv.Comp, dt.datetime]] | None = None,
    ) -> list[Event]:
        i_s, e_s, dt_s = self.itail(n=n, where_i=where_i, where_dt=where_dt)
        return [
            replace(e, i=i) for e, i in zip(e_s, i_s)
        ]

@dataclass(frozen=True)
class EventQueueSQLite(
    EventQueue,
    kv.SQLiteQueue[int, float, bytes, dt.datetime, Event]
):
    """
    >>> import time
    >>> j, k, l = ("j", "k", "l")
    >>> @dataclass(frozen=True)
    ... class EventI(Event):
    ...     i: int | None = Event.i_field()
    ...     type: Literal['A'] = Event.type_field("A")
    >>> q = EventQueueSQLite.new(TEST_DB, EventI.parser())
    >>> q.head(1)
    []
    >>> q.tail(1)
    []
    >>> v = q.append(EventI(k))
    >>> assert v == EventI(k) == q.head(1)[0]
    >>> assert ([1], [EventI(k)]) == q.ihead(1)[:2]
    >>> time.sleep(0.1)
    >>> dt = datetime.datetime.now()
    >>> time.sleep(0.1)
    >>> _ = q.append(EventI(j))
    >>> _ = q.append(EventI(l))
    >>> q.tail(n=3, where_dt=[("gt", dt)])
    [EventI(obj='l'), EventI(obj='j')]
    >>> q.head(n=3, where_dt=[("gt", dt)])
    [EventI(obj='j'), EventI(obj='l')]
    """

    @classmethod
    def new(
        cls,
        db: str,
        parser: Type[EventParser],
        table: str = "Events",
    ) -> EventQueueSQLite:
        self = cls(config=kv.SQLiteQueueConfig(
            db=db, name=table
        ), parser=parser)
        return cast(EventQueueSQLite, self.prepare())


__all__ = [
    "EventParser",
    "Event",
    "EventQueue",
    "EventQueueSQLite",
]