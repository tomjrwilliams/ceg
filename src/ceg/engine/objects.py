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

from .events import *
from .snaps import *
from .values import *

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
    >>> Object.load_json(a.dump_bytes(), ObjectParserAB)
    ObjectA(path='')
    >>> Object.load_json(b.dump_bytes(), ObjectParserAB) 
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
    def load_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[ObjectParser]
    ) -> Object:
        parser: ObjectParser = into.ta().validate_json(data)
        return parser.obj

    # -------------
    
    # NOTE: listen to any event for these objects
    def depends(self) -> list[Dependency]: ...

    # NOTE: return the handlers to run for a given event
    # versus having to init all up front, even if not used
    def handlers(self, event: Event) -> list[Handler]: ...

    # NOTE: override with more specific type return
    # NOTE: contains last processed event
    def snap(self) -> Snap: ...

    # NOTE: override with more specific type return
    def value(self) -> Value: ...

    # up through to root object as needed
    def publish(self, event: Event) -> None: ...

    # -------------
    

@dataclass(frozen=True)
class Dependency:
    obj: Object
    depends: Object
    active: bool = True
    parent: bool = False
    ancestor: bool = False
    events: tuple[str, ...] | None = None
    # NOTE: types, used to filter without .handlers()

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
    "Dependency",
    "Handler",
]