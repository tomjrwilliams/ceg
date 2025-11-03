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

from frozendict import frozendict

from ..apis import kv

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
    ...     type: Literal['A'] = Event.type_field("A")
    >>> @dataclass(frozen=True)
    ... class EventB(Event):
    ...     b: float
    ...     type: Literal['B'] = Event.type_field("B")
    >>> @dataclass(frozen=True)
    ... class EventParserAB(EventParser):
    ...     event: EventA | EventB = Event.union_field()
    >>> e_a, e_b = EventA(0, "a", 1), EventB(1, "b", 1.)
    >>> _ = EventParserAB(e_a) # init parser type
    >>> Event.parse_json(e_a.dump_bytes(), EventParserAB)
    EventA(0, obj='a', a=1)
    >>> Event.parse_json(e_b.dump_bytes(), EventParserAB) 
    EventB(1, obj='b', b=1.0)
    """
    i: int # NOTE: event id (ordered globally)
    obj: str # NOTE: object path

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

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

    # TODO: or return ID?
    def append_json(self, q: kv.Queue) -> None:
        return q.append(self.dump_bytes())

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class EventQueue(
    kv.QueueTimestampAsFloat[int, bytes, Event]
):
    
    type_id: Type[int] = int
    type_val_i: Type[bytes] = bytes
    type_val_o: Type[Event] = Event

    parser: Type[EventParser] | None = None

    def encode_val_type(self, t: Type[Event]) -> Type[bytes]:
        return bytes

    def encode_val(self, v: Event) -> bytes:
        return v.dump_bytes()

    def decode_val(self, v: bytes) -> Event:
        assert self.parser is not None, self
        return Event.parse_json(v, into=self.parser)

# TODO: EventQueueSQlite

__all__ = [
    "EventParser",
    "Event",
    "EventQueue",
]