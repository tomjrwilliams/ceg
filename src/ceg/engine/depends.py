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

from ..apis import kv

# TODO: string wrapper for path types

@dataclass(frozen=True)
class Dependency:
    obj: str
    depends: str
    active: bool = True
    parent: bool = False
    ancestor: bool = False
    events: tuple[str, ...] | None = None
    # NOTE: types, used to filter without .handlers()

    def key(self) -> str:
        return f"{self.obj},{self.depends}"

    # forward proofing if we eventually have diff fields
    @property
    def type(self) -> str: ...

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
        type: TDependency,
    ) -> TDependency:
        return type.parse_json(data, type.parser())

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
    kv.KeyValStrKeyIBytesValI[int, Dependency]
):
    
    type_id: Type[int] = int
    type_val_o: Type[Dependency] = Dependency

    parser: Type[DependencyParser] | None = None

    def encode_val_type(self, t: Type[Dependency]) -> Type[bytes]:
        return bytes

    def encode_val(self, v: Dependency) -> bytes:
        return v.dump_bytes()

    def decode_val(self, v: bytes) -> Dependency:
        assert self.parser is not None, self
        return Dependency.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class DependencyKeyValSQLite(
    DependencyKeyVal,
    kv.SQLiteKeyVal[int, str, bytes, str, Dependency]
):
    """
    >>> k, j, updated = ("k", "j", [])
    >>> kv = DependencyKeyValSQLite.new(
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
        cls, db: str, table: str = "Depends"
    ) -> DependencyKeyValSQLite:
        return cls(config=kv.SQLiteKeyValConfig(
            db=db, name=table
        ))

__all__ = [
    "Dependency",
    "DependencyParser",
    "DependencyKeyVal",
]