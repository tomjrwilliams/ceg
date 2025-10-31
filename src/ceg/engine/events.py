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
    >>> e_a, e_b = EventA("a", 1), EventB("b", 1.)
    >>> _ = EventParserAB(e_a) # init parser type
    >>> Event.load_json(e_a.dump_bytes(), EventParserAB)
    EventA(obj='a', a=1)
    >>> Event.load_json(e_b.dump_bytes(), EventParserAB) 
    EventB(obj='b', b=1.0)
    """
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
    def load_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[EventParser]
    ) -> Event:
        parser: EventParser = into.ta().validate_json(data)
        return parser.event

__all__ = [
    "EventParser",
    "Event",
]