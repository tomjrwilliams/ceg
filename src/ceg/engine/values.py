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

from ..apis import type_vars
from ..apis import kv

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

    # TODO: parse_json_as, parse_json_as_self
    
    @classmethod
    def get_json(
        cls,
        key: str,
        kv: kv.KeyVal # TODO: type narrow?
    ) -> str | bytes | bytearray | None:
        # assert isinstance(kv, DependencyKeyVal)
        return kv.get(key)

    def set_json(
        self,
        kv: kv.KeyVal # TODO: type narrow?
    ) -> None:
        # or return id?
        return kv.set(self.obj, self.dump_bytes())

@dataclass(frozen=True)
class ValueKeyVal(
    kv.KeyValStrKeyI[int, type_vars.ValI, Value]
):
    
    type_id: Type[int] = int
    type_val_o: Type[Value] = Value

    parser: Type[ValueParser] | None = None

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class JSONValueKeyVal(ValueKeyVal[bytes]):

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

# TODO: JSONValueKeyValSQlite per value type

# TODO: numpyValueKeyVal, ..sqlite etc.

__all__ = [
    "ValueParser",
    "Value",
    "ValueKeyVal",
    "JSONValueKeyVal",
]