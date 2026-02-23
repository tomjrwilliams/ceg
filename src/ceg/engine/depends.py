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
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from pydantic import (
    TypeAdapter,
    BaseModel,
    Field,
)
from frozendict import frozendict

from ..apis import kv, assert_is

import os
TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

# TODO: string wrapper for path types

@dataclass(frozen=True)
class Dependency:
    obj: str
    depends: str
    direct: bool = True
    # direct is used to indicate if triggers, or not
    # as if not, then used to indicate who creates the value, for traversign event history when debugging

    # TODO: so need to fill out dependenices for non directly created (ie. parent often)
    
    # NOTE: event types, used to filter without .handlers()
    events: tuple[str, ...] | None = None

    # NOTE: forward proofing if we eventually have diff fields
    type: Literal["Dependency"] = "Dependency"

    def key(self) -> str:
        return f"{self.obj},{self.depends}"

    # @property
    # def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[DependencyParser]:
        return dependency_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[DependencyParser]
    ) -> Dependency:
        parser: DependencyParser = into.ta().validate_json(data)
        return parser.dep
    
    @classmethod
    def parse_json_as(
        cls,
        data: str | bytes | bytearray,
        type: Type[TDependency],
    ) -> TDependency:
        v = type.parse_json(data, type.parser())
        return assert_is(v, type)

    @classmethod
    def parse_json_as_self(
        cls,
        data: str | bytes | bytearray,
    ) -> Dependency: ...
    # TODO: call parse_json_as, override type return

    @classmethod
    def get(
        cls,
        key: str,
        kv: DependencyKeyVal
    ) -> Dependency | None:
        # TODO: iget
        return kv.get(key)

    def set(
        self,
        kv: DependencyKeyVal
    ) -> Dependency:
        # TODO: iset
        return kv.set(self.key(), self)

TDependency = TypeVar("TDependency", bound=Dependency)

@dataclass(frozen=True)
class DependencyParser:
    dep: Dependency

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)

@functools.lru_cache(maxsize=128, typed=True)
def dependency_parser(
    cls: Type[Dependency]
) -> Type[DependencyParser]:
    return dataclass(frozen=True)(type(
        cls.__name__ + "Parser",
        (DependencyParser,),
        {
            "dep": Field(discriminator="type"),
            "__annotations__": {"dep": cls}
        }
    ))

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class DependencyKeyVal(
    kv.KeyValStrKeyIOBytesValI[int, Dependency]
):
    
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes] = bytes
    type_key_o: Type[str] = str
    type_val_o: Type[Dependency] = Dependency

    parser: Type[DependencyParser] | None = None

    def encode_val_type(self, t: Type[Dependency]) -> Type[bytes]:
        return bytes

    def encode_val(self, k: str, v: Dependency) -> bytes:
        return v.dump_bytes()

    def decode_val(self, k: str, v: bytes) -> Dependency:
        assert self.parser is not None, self
        return Dependency.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class DependencyKeyValSQLite(
    DependencyKeyVal,
    kv.SQLiteKeyVal[int, str, bytes, str, Dependency]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> kv = DependencyKeyValSQLite.new(
    ...     TEST_DB,
    ...     on_update = lambda k, v: updated.append((k, v)),
    ...     parser=Dependency.parser()
    ... )
    >>> kv.get(k)
    >>> kk, jj, jk, kj = Dependency(k, k), Dependency(j, j), Dependency(j, k), Dependency(k, j)
    >>> v = kv.set(k, kk)
    >>> assert v == kk == kv.get(k), kv.get(k)
    >>> assert (1, kk) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, jj)
    >>> v = kv.set(k, kj)
    >>> assert (2, jj) == kv.iget(j)
    >>> assert (1, kj) == kv.iget(k)
    >>> assert updated == [(k, kj)]
    """

    @classmethod
    def new(
        cls, 
        db: str, 
        parser: Type[DependencyParser],
        table: str = "Depends",
        on_update: Callable[[str, Dependency], None] | None = None
    ) -> DependencyKeyValSQLite:
        self = cls(
            config=kv.SQLiteKeyValConfig(
                db=db, name=table
            ), 
            on_update=on_update,
            parser=parser
        )
        return cast(DependencyKeyValSQLite, self.prepare())

__all__ = [
    "Dependency",
    "DependencyParser",
    "DependencyKeyVal",
    "DependencyKeyValSQLite",
]