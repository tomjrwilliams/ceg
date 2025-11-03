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

from .events import *
from .snaps import *
from .values import *
from .depends import *

T = TypeVar("T")

@dataclass(frozen=True)
class ObjectParser:
    obj: Object

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)

@functools.lru_cache(maxsize=128, typed=True)
def object_parser(cls: Type[Object]) -> Type[ObjectParser]:
    return dataclass(frozen=True)(type(
        cls.__name__ + "Parser",
        (ObjectParser,),
        {
            "obj": Field(discriminator="type"),
            "__annotations__": {"obj": cls}
        }
    ))

@dataclass(frozen=True)
class Object:
    """
    >>> @dataclass(frozen=True)
    ... class ObjectA(Object):
    ...     type: Literal['A'] = Object.type_field("A")
    >>> @dataclass(frozen=True)
    ... class ObjectB(Object):
    ...     parent: ObjectA
    ...     type: Literal['B'] = Object.type_field("B")
    >>> @dataclass(frozen=True)
    ... class ObjectParserAB(ObjectParser):
    ...     obj: ObjectA | ObjectB = Object.union_field()
    >>> a = ObjectA("")
    >>> b = ObjectB("", a)
    >>> _ = ObjectParserAB(a) # init parser type
    >>> Object.parse_json(a.dump_bytes(), ObjectParserAB)
    ObjectA(path='')
    >>> Object.parse_json(b.dump_bytes(), ObjectParserAB) 
    ObjectB(path='', parent=ObjectA(path=''))
    """
    path: str
    parent: Object | None = Field(default=None, repr=False)

    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[ObjectParser]:
        return object_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[ObjectParser]
    ) -> Object:
        parser: ObjectParser = into.ta().validate_json(data)
        return parser.obj

    # -------------

    @overload
    def inherit(
        self,
        field: str,
        type: Type[T],
        strict: Literal[True],
    ) -> T: ...

    @overload
    def inherit(
        self,
        field: str,
        type: Type[T],
        strict: bool = False,
    ) -> T | None: ...

    def inherit(
        self,
        field: str,
        type: Type[T],
        strict: bool = False,
    ) -> T | None:
        if hasattr(self, field):
            v = getattr(self, field)
            assert isinstance(v, type), (v, type)
            return v
        elif self.parent is not None:
            return self.parent.inherit(field, type, strict)
        if strict:
            raise ValueError(field, self)
        return None

    # up through to root object as needed
    def publish(self, event: Event) -> None:
        q = self.event_queue()
        return event.append_json(q)
        
    # -------------
    # NOTE: roughly in order of likelihood of override

    # NOTE: as for snap, likely not often overridden (root)
    def event_queue(self) -> EventQueue: ...

    # NOTE: also generally not likely often overriden
    def dependency_key_val(self) -> DependencyKeyVal: ...

    # TODO: generic one implemented at root that has a switch on fields on object for backend type and params (eg. type=sqlite, db=..., table=...)
    # and io as SnapRawIO (bytes i, bytes o)
    # but more likely local IO overriden
    def snap_key_val(self) -> SnapKeyVal: ...

    # NOTE: override with more specific type return
    # NOTE: contains last processed event
    # doesnt take key_val: just call
    def get_snap(self) -> Snap: ...

    # TODO: tbc return type? id?
    def set_snap(self) -> Snap: ...

    # NOTE: as for snap, but likely more object specific, migth be key_val subtype (eg. chunked), depending on value type (get_value then calls value_key_val)
    def value_key_val(self) -> ValueKeyVal: ...

    # NOTE: override with more specific type return
    def get_value(self) -> Value: ...

    # TODO: tbc return type? id?
    def set_value(self) -> Value: ...

    # NOTE: listen to any event for these objects
    def depends(self) -> list[Dependency]: ...

    # NOTE: return the handlers to run for a given event
    # versus having to init all up front, even if not used
    def handlers(self, event: Event) -> list[Handler]: ...

    # -------------


@dataclass(frozen=True)
class Handler:
    
    def handle(
        self,
        obj: Object,
        event: Event,
    ) -> None: ... # publish via object itself

__all__ = [
    "ObjectParser",
    "Object",
    "Handler",
]