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

from ..apis import kv
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
    ...     type: Literal['A'] = Snap.type_field("A")
    >>> @dataclass(frozen=True)
    ... class SnapB(Snap):
    ...     b: float
    ...     type: Literal['B'] = Snap.type_field("B")
    >>> @dataclass(frozen=True)
    ... class SnapParserAB(SnapParser):
    ...     snap: SnapA | SnapB = Snap.union_field()
    >>> a, b = SnapA("a", 1), SnapB("b", 1.)
    >>> _ = SnapParserAB(a) # init parser type
    >>> Snap.parse_json(a.dump_bytes(), SnapParserAB)
    SnapA(obj='a', a=1)
    >>> Snap.parse_json(b.dump_bytes(), SnapParserAB) 
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
        into: Type[SnapParser]
    ) -> Snap:
        parser: SnapParser = into.ta().validate_json(data)
        return parser.snap

    @classmethod
    def get(
        cls,
        key: str,
        kv: SnapKeyVal,
    ) -> Snap | None:
        # TODO: iget
        return kv.get(key)

    def set(
        self,
        kv: SnapKeyVal
    ) -> Snap:
        # TODO: iset
        return kv.set(self.obj, self)

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class SnapKeyVal(
    kv.KeyValStrKeyIBytesValI[int, Snap]
):
    
    type_id: Type[int] = int
    type_val_o: Type[Snap] = Snap

    parser: Type[SnapParser] | None = None

    def encode_val_type(self, t: Type[Snap]) -> Type[bytes]:
        return bytes

    def encode_val(self, v: Event) -> bytes:
        return v.dump_bytes()

    def decode_val(self, v: bytes) -> Snap:
        assert self.parser is not None, self
        return Snap.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class SnapKeyValSQLite(
    SnapKeyVal,
    kv.SQLiteKeyVal[int, str, bytes, str, Snap]
):
    """
    >>> k, j, updated = ("k", "j", [])
    >>> kv = SnapKeyValSQLite.new(
    ...     TEST_DB,
    ...     on_update = lambda k, v: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k)
    >>> v = kv.set(k, 0.0)
    >>> assert v == 0.0 == kv.get(k), kv.get(k)
    >>> assert (1, 0.0) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, 1.0)
    >>> v = kv.set(k, 2.0)
    >>> assert (2, 1.0) == kv.iget(j)
    >>> assert (1, 2.0) == kv.iget(k)
    >>> assert updated == [(k, 2.0)]
    """

    @classmethod
    def new(
        cls, db: str, table: str = "Snaps"
    ) -> SnapKeyValSQLite:
        return cls(config=kv.SQLiteKeyValConfig(
            db=db, name=table
        ))

__all__ = [
    "SnapParser",
    "Snap",
    "SnapKeyVal",
    "SnapKeyValSQLite",
]