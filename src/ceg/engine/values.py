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

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

from ..apis import type_vars
from ..apis import kv

import os
TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

@dataclass(frozen=True)
class ValueParser:
    value: Value

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)

@functools.lru_cache(maxsize=128, typed=True)
def value_parser(cls: Type[Value]) -> Type[ValueParser]:
    return dataclass(frozen=True)(type(
        cls.__name__ + "Parser",
        (ValueParser,),
        {
            "value": Field(discriminator="type"),
            "__annotations__": {"value": cls}
        }
    ))


@dataclass(frozen=True)
class Value:
    """
    >>> @dataclass(frozen=True)
    ... class ValueA(Value):
    ...     a: int
    ...     type: Literal['A'] = Value.type_field("A")
    >>> @dataclass(frozen=True)
    ... class ValueB(Value):
    ...     b: float
    ...     type: Literal['B'] = Value.type_field("B")
    >>> @dataclass(frozen=True)
    ... class ValueParserAB(ValueParser):
    ...     value: ValueA | ValueB = Value.union_field()
    >>> a, b = ValueA("a", 1), ValueB("b", 1.)
    >>> _ = ValueParserAB(a) # init parser type
    >>> Value.parse_json(a.dump_bytes(), ValueParserAB)
    ValueA(obj='a', a=1)
    >>> Value.parse_json(b.dump_bytes(), ValueParserAB) 
    ValueB(obj='b', b=1.0)
    """
    obj: str # NOTE: object path

    # NOTE: can have other fields for json-type value objects
    # NOTE: meta (calendar etc. for time series) should appear in snap instead

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[ValueParser]:
        return value_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[ValueParser]
    ) -> Value:
        parser: ValueParser = into.ta().validate_json(data)
        return parser.value
    
    @classmethod
    def get(
        cls,
        key: str,
        kv: ValueKeyVal,
    ) -> Value | None:
        # TODO: iget
        return kv.get(key)

    def set(
        self,
        kv: ValueKeyVal,
    ) -> Value:
        # TODO: iset
        return kv.set(self.obj, self)

@dataclass(frozen=True)
class ValueKeyVal(
    kv.KeyValStrKeyIO[int, type_vars.ValI, Value]
):
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes] = bytes
    type_key_o: Type[str] = str
    type_val_o: Type[Value] = Value

    parser: Type[ValueParser] | None = None

    # TODO: get and set are generic, they just need the relevant encode and decode methods set in the types passed to the relevant key val for the value type

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class ValueKeyValJSON(ValueKeyVal[bytes]):

    type_val_i: Type[bytes] = bytes
    
    def encode_val_type(self, t: Type[Value]) -> Type[bytes]:
        return bytes

    def encode_val(self, v: Value) -> bytes:
        # TODO: extract any non json vals?
        # ie. if store numpy array directly?
        return v.dump_bytes()

    def decode_val(self, v: bytes) -> Value:
        # TODO: extract any non json vals?
        # ie. if store numpy array directly?
        assert self.parser is not None, self
        return Value.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class ValueKeyValJSONSQLite(
    ValueKeyValJSON,
    kv.SQLiteKeyVal[int, str, bytes, str, Value]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> @dataclass(frozen=True)
    ... class ValueEx(Value):
    ...     a: int
    ...     type: Literal['Ex'] = Value.type_field("Ex")
    >>> kv = ValueKeyValJSONSQLite.new(
    ...     TEST_DB, 
    ...     parser = ValueEx.parser(),
    ...     on_update = lambda k, v: updated.append((k, v))
    ... )
    >>> kv.get(k)
    >>> v = kv.set(k, ValueEx(k, 0.0))
    >>> assert v == ValueEx(k, 0.0) == kv.get(k), kv.get(k)
    >>> assert (1, ValueEx(k, 0.0)) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, ValueEx(j, 1.0))
    >>> v = kv.set(k, ValueEx(k, 2.0))
    >>> assert (2, ValueEx(j, 1.0)) == kv.iget(j)
    >>> assert (1, ValueEx(k, 2.0)) == kv.iget(k)
    >>> assert updated == [(k, ValueEx(k, 2.0))]
    """

    @classmethod
    def new(
        cls,
        db: str, 
        parser: Type[ValueParser],
        table: str = "ValuesJSON",
        on_update: Callable[[str, Value], None] | None = None
    ) -> ValueKeyValJSONSQLite:
        self = cls(config=kv.SQLiteKeyValConfig(
            db=db, name=table
        ), on_update=on_update, parser=parser)
        return cast(ValueKeyValJSONSQLite, self.prepare())

# TODO: numpy chunked, etc.
# time series? subtype of numpy, where either store 2d with a start point in snap, or store dense (flag in snap)
# TODO: numpyValueKeyVal, ..sqlite etc.

__all__ = [
    "ValueParser",
    "Value",
    "ValueKeyVal",
    "ValueKeyValJSON",
    "ValueKeyValJSONSQLite",
]