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
    >>> Value.load_json(a.dump_bytes(), ValueParserAB)
    ValueA(obj='a', a=1)
    >>> Value.load_json(b.dump_bytes(), ValueParserAB) 
    ValueB(obj='b', b=1.0)
    """
    obj: str # NOTE: object path

    # NOTE: value sub type fields for eg. time series, etc. config / meta information (calendar, etc.)
    # store both the json via key_val and the actual value (s?) via key_val_chunked

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
    def load_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[ValueParser]
    ) -> Value:
        parser: ValueParser = into.ta().validate_json(data)
        return parser.value

# NOTE: sub types implement their respective set / get methods in such a way that they can run just given the object
# ie. have methods to return a backend given the relevant parent / root objects (eg. env variables: db paths, etc.)

# ie. switch on the backend from a flag in the env / root object (or object itself?) that is used by the object itself to decide what to load (given the other env / root object fields eg. db paths, etc.)

__all__ = [
    "ValueParser",
    "Value",
]