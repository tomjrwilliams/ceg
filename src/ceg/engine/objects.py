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
class Handler:
    
    def handle(
        self,
        obj: Object,
        event: Event,
    ) -> None: ... # publish via object itself

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
    def publish(self, event: Event) -> Event:
        q = self.event_queue()
        return event.append(q)
        
    # -------------
    # NOTE: roughly in order of likelihood of override

    # NOTE: as for snap, likely not often overridden (root)
    def event_queue(self) -> EventQueue: ...

    # NOTE: also generally not likely often overriden
    def dependency_key_val(self) -> DependencyKeyVal: ...

    def obj_key_val(self) -> ObjectKeyVal: ...

    def set_obj(self) -> Object: ...

    # get current stored, for differencing
    def get_obj(self) -> Object: ...

    # TODO: generic one implemented at root that has a switch on fields on object for backend type and params (eg. type=sqlite, db=..., table=...)
    # and io as SnapRawIO (bytes i, bytes o)
    # but more likely local IO overriden
    def snap_key_val(self) -> SnapKeyVal: ...

    # NOTE: override with more specific type return
    # NOTE: contains last processed event
    # doesnt take key_val: just call
    # TODO: only here because needs object, so not in snap - maybe move all to a separate utils / lib module?
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

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class ObjectKeyVal(
    kv.KeyValStrKeyIBytesValI[int, Object]
):
    
    type_id: Type[int] = int
    type_val_o: Type[Object] = Object

    parser: Type[ObjectParser] | None = None

    def encode_val_type(self, t: Type[Object]) -> Type[bytes]:
        return bytes

    def encode_val(self, v: Object) -> bytes:
        return v.dump_bytes()

    def decode_val(self, v: bytes) -> Object:
        assert self.parser is not None, self
        return Object.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class ObjectKeyValSQLite(
    ObjectKeyVal,
    kv.SQLiteKeyVal[int, str, bytes, str, Object]
):
    """
    >>> k, j, updated = ("k", "j", [])
    >>> kv = ObjectKeyValSQLite.new(
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
        cls, db: str, table: str = "Objects"
    ) -> ObjectKeyValSQLite:
        return cls(config=kv.SQLiteKeyValConfig(
            db=db, name=table
        ))

__all__ = [
    "Handler",
    "ObjectParser",
    "Object",
    "ObjectKeyVal",
    "ObjectKeyValSQLite",
]