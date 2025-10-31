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

from .events import *

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
    return dataclass(frozen=True)(type(
        cls.__name__ + "Parser",
        (SnapParser,),
        {
            "snap": Field(discriminator="type"),
            "__annotations__": {"snap": cls}
        }
    ))

@dataclass(frozen=True)
class Snap:
    """
    >>> @dataclass(frozen=True)
    ... class SnapA(Snap):
    ...     a: int
    ...     type: Literal['A'] = Snap.typfield("A")
    >>> @dataclass(frozen=True)
    ... class SnapB(Snap):
    ...     b: float
    ...     type: Literal['B'] = Snap.typfield("B")
    >>> @dataclass(frozen=True)
    ... class SnapParserAB(SnapParser):
    ...     snap: SnapA | SnapB = Snap.union_field()
    >>> a, b = SnapA("a", 1), SnapB("b", 1.)
    >>> _ = SnapParserAB(a) # init parser type
    >>> Snap.load_json(a.dump_bytes(), SnapParserAB)
    SnapA(obj='a', a=1)
    >>> Snap.load_json(b.dump_bytes(), SnapParserAB) 
    SnapB(obj='b', b=1.0)
    """
    event: Event
    obj: str # NOTE: object path

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def typfield(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[SnapParser]:
        return snap_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def load_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[SnapParser]
    ) -> Snap:
        parser: SnapParser = into.ta().validate_json(data)
        return parser.snap

__all__ = [
    "SnapParser",
    "Snap",
]