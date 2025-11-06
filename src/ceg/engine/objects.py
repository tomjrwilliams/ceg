from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    Any,
    overload,
    cast,
    final,
)
import functools

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass
from dataclasses import replace, fields

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

import os
TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

T = TypeVar("T")

@dataclass(frozen=True)
class Handler:
    
    def handle(
        self,
        event: Event,
        obj: Object,
        domain: Domain,
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
    ...     parent: str | None = Object.parent_field()
    ...     type: Literal['A'] = Object.type_field("A")
    >>> @dataclass(frozen=True)
    ... class ObjectB(Object):
    ...     parent: ObjectA | str
    ...     type: Literal['B'] = Object.type_field("B")
    >>> @dataclass(frozen=True)
    ... class ObjectParserAB(ObjectParser):
    ...     obj: ObjectA | ObjectB = Object.union_field()
    >>> a = ObjectA("a")
    >>> b = ObjectB("b", a)
    >>> _ = ObjectParserAB(a) # init parser type
    >>> Object.parse_json(a.dump_bytes(), ObjectParserAB)
    ObjectA(path='a')
    >>> Object.parse_json(b.dump_bytes(), ObjectParserAB) 
    ObjectB(path='b', parent='a')
    """
    path: str
    parent: str | Object | None

    # NOTE: string to avoid duplication on insert
    
    @property
    def type(self) -> str: ...
    
    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parent_field(
        cls,
        default: Object | None = None, 
        repr: bool = False
    ):
        return Field(default=default, repr=repr)

    @classmethod
    def parser(cls) -> Type[ObjectParser]:
        return object_parser(cls)

    def dump_bytes(self) -> bytes:
        parser = self.parser()
        if isinstance(self.parent, Object):
            self = replace(self, parent=self.parent.path)
        return parser(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[ObjectParser]
    ) -> Object:
        parser: ObjectParser = into.ta().validate_json(data)
        return parser.obj

    def check(self):
        assert self.type == type(self).__name__, self
        return self

    # -------------

    @overload
    def inherit(
        self,
        field: str,
        type: Type[T],
        domain: frozendict[str, Object],
        strict: Literal[True],
    ) -> T: ...

    @overload
    def inherit(
        self,
        field: str,
        type: Type[T],
        domain: frozendict[str, Object],
        strict: bool = False,
    ) -> T | None: ...

    def inherit(
        self,
        field: str,
        type: Type[T],
        domain: frozendict[str, Object],
        strict: bool = False,
    ) -> T | None:
        if hasattr(self, field):
            v = getattr(self, field)
            assert isinstance(v, type), (v, type)
            return v
        elif self.parent is not None:
            if isinstance(self.parent, str):
                parent = domain.get(self.parent)
            else:
                parent = self.parent
            if parent is None:
                raise ValueError((self, parent))
            return parent.inherit(
                field, type, domain, strict=strict
            )
        if strict:
            raise ValueError(field, self)
        return None

    def diff(self, new: Object) -> ObjectDiff | None:
        return ObjectDiff.new(self, new)

    # up through to root object as needed
    def publish(self, event: Event, domain: Domain) -> Event:
        q = self.event_queue(domain)
        return event.append(q)

    # TODO: separate classmethods that arent override for getting obj, snap, value, etc.

    # NOTE: not classmethod so can be override more specifically
    def get_obj(self, domain: Domain) -> Object | None: 
        return self.obj_key_val(domain).get(self.path)

    def set_obj(self, domain: Domain) -> Object:
        return self.obj_key_val(domain).set(self.path, self)

    # NOTE: not a classmethod, so can be overriden with more specific type by child
    def get_snap(self, domain: Domain) -> Snap | None:
        return self.snap_key_val(domain).get(self.path)

    def set_snap(self, snap: Snap, domain: Domain) -> Snap:
        return self.snap_key_val(domain).set(self.path, snap)

    # NOTE: not classmethod so can be override more specifically
    def get_value(self, domain: Domain) -> Value | None:
        return self.value_key_val(domain).get(self.path)

    def set_value(self, value: Value, domain: Domain) -> Value:
        return self.value_key_val(domain).set(self.path, value)
        
    # NOTE: value change isnt emit by set by default
    # rather, up to object to decide if updating their own value, to emit the event (as part of the handle)
    # eg. for append / prepend - more specific than just update

    # -------------
    # NOTE: roughly in order of likelihood of override

    # NOTE: as for snap, likely not often overridden (root)
    def event_queue(self, domain: Domain) -> EventQueue: ...

    # NOTE: also generally not likely often overriden
    def dependency_key_val(self, domain: Domain) -> DependencyKeyVal: ...

    def obj_key_val(self, domain: Domain) -> ObjectKeyVal: ...

    # TODO: generic one implemented at root that has a switch on fields on object for backend type and params (eg. type=sqlite, db=..., table=...)
    # and io as SnapRawIO (bytes i, bytes o)
    # but more likely local IO overriden
    def snap_key_val(self, domain: Domain) -> SnapKeyVal: ...

    # NOTE: as for snap, but likely more object specific, migth be key_val subtype (eg. chunked), depending on value type (get_value then calls value_key_val)
    def value_key_val(self, domain: Domain) -> ValueKeyVal: ...

    # NOTE: listen to any event for these objects
    def depends(self, domain: Domain) -> list[Dependency]: ...

    # NOTE: return the handlers to run for a given event
    # versus having to init all up front, even if not used
    def handlers(self, event: Event, domain: Domain) -> list[Handler]: ...

    # -------------

# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class ObjectKeyVal(
    kv.KeyValStrKeyIOBytesValI[int, Object]
):
    
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes] = bytes
    type_key_o: Type[str] = str
    type_val_o: Type[Object] = Object

    parser: Type[ObjectParser] | None = None

    def encode_val_type(self, t: Type[Object]) -> Type[bytes]:
        return bytes

    def encode_val(self, k: str, v: Object) -> bytes:
        return v.dump_bytes()

    def decode_val(self, k: str, v: bytes) -> Object:
        assert self.parser is not None, self
        return Object.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class ObjectKeyValSQLite(
    ObjectKeyVal,
    kv.SQLiteKeyVal[int, str, bytes, str, Object]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> @dataclass(frozen=True)
    ... class ObjectEx(Object):
    ...     parent: Object | None = Object.parent_field()
    ...     type: Literal['Ex'] = Object.type_field("Ex")
    >>> kv = ObjectKeyValSQLite.new(
    ...     TEST_DB,
    ...     parser=ObjectEx.parser(),
    ...     on_update = lambda k, v: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k)
    >>> v = kv.set(k, ObjectEx(k))
    >>> assert v == ObjectEx(k) == kv.get(k), kv.get(k)
    >>> assert (1, ObjectEx(k)) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, ObjectEx(j))
    >>> v = kv.set(k, ObjectEx(l))
    >>> assert (2, ObjectEx(j)) == kv.iget(j)
    >>> assert (1, ObjectEx(l)) == kv.iget(k)
    >>> assert updated == [(k, ObjectEx(l))]
    """

    @classmethod
    def new(
        cls, db: str, 
        parser: Type[ObjectParser],table: str = "Objects",
        on_update: Callable[[str, Object], None] | None = None
    ) -> ObjectKeyValSQLite:
        return cls(config=kv.SQLiteKeyValConfig(
            db=db, name=table
        ), on_update=on_update, parser=parser)

@dataclass(frozen=True)
class FieldDiff:
    key: str
    old: Any | None
    new: Any | None
    present: tuple[bool, bool] # old, new

@dataclass(frozen=True)
class ObjectDiff:
    types: tuple[str, str]
    fields: dict[str, FieldDiff]

    @classmethod
    def new(
        cls, old: Object, new: Object,
    ) -> ObjectDiff | None:
        flds = {}
        for fld in fields(old) + fields(new):
            k = fld.name
            if k in flds:
                continue
            has_old = hasattr(old, k)
            has_new = hasattr(new, k)
            v_old = getattr(old, k, None)
            v_new = getattr(new, k, None)
            if v_old is None and v_new is None:
                continue
            if (has_old or has_new) and (v_old != v_new):
                flds[k] = FieldDiff(
                    k,
                    v_old,
                    v_new,
                    (has_old, has_new)
                )
        return ObjectDiff(
            (
                type(old).__name__,
                type(new).__name__,
            ),
            flds
        )

@dataclass(frozen=True)
class Domain:
    root: Object # snap is used for domain on_run trigger
    objects: dict[str, Object] = Field(
        default_factory=dict
    )
    depends: dict[str, dict[str, Dependency]] = Field(
        default_factory=dict
    )
    
    @property
    def event_parser(self) -> Type[EventParser]:
        return DefaultEventParser

    @property
    def depends_parser(self) -> Type[DependencyParser]:
        return DependencyParser

    @property
    def object_parser(self) -> Type[ObjectParser]: ...

    @property
    def snap_parser(self) -> Type[SnapParser]: ...

    @property
    def value_parser(self) -> Type[ValueParser]: ...

    def with_obj(self, obj: Object) -> Domain:
        obj = obj.check()
        assert obj.path not in self.objects, obj.path
        # TODO: depends
        # TODO: add parents if not given
        return replace(
            self, objects={
                **self.objects,
                **{obj.path: obj}
            }
        )
    
    # NOTE: we dont need to diff the domain itself
    # if we add objects then their depends will be picked up on sync, and we know to iter the whole queue again, regardless of the domains own latest
    def sync(self) -> Domain:
        
        # TODO: or object.value not snap?
        snap = self.root.get_snap(self)
        if snap is None:
            curr = None
            curr_i = -1
        else:
            curr = snap.event
            assert curr.i is not None, curr
            curr_i = curr.i

        q = self.root.event_queue(self)

        events: list[Event]
        history: list[Event] | None = None

        for i_path, triggers in self.depends.items():
            for o_path, dep in triggers.items():
                obj = self.objects.get(o_path)
                assert obj is not None, (o_path)

                if not dep.active:
                    dep = dep.set(
                        obj.dependency_key_val(self)
                    )
                    continue

                dep_prev = dep.get(
                    dep.key(),
                    obj.dependency_key_val(self)
                )

                events = []
                if dep_prev is None:
                    history = history or q.head_with_i(
                        n=-1,
                        where_i = [("le", curr_i)]
                        # TODO: head = from first?
                    )
                    events = [
                        e for e in history
                        if e.obj == i_path
                    ]
                elif dep.events is not None:
                    blob_events = dep_prev.events or []
                    e_types = [
                        e_type for e_type in dep.events
                        if e_type not in blob_events
                    ]
                    if len(e_types):
                        history = history or q.head_with_i(
                            n=-1,
                            where_i = [("le", curr_i)]
                            # TODO: head = from first?
                        )
                        events = [
                            e for e in history
                            if e.obj == i_path
                            and e.type in e_types    
                        ]

                for e in events:
                    handlers = obj.handlers(e, self)
                    for handler in handlers:
                        handler.handle(e, obj, self)

                dep = dep.set(
                    obj.dependency_key_val(self)
                )

        e_sync = self.root.publish(DomainSync(
            self.root.path,
        ), self)
        assert e_sync.i is not None, e_sync
        
        for k, obj_new in self.objects.items():
            obj_old = obj_new.get_obj(self)
            if obj_old is None:
                e_new = self.root.publish(ObjectNew(k), self)
                assert e_new.i is not None, e_new
            else:
                diff = obj_old.diff(obj_new)
                if diff is None:
                    continue
                e_chg = self.root.publish(
                    ObjectChange(k, diff), self
                )
                assert e_chg.i is not None, e_chg

        events = q.head_with_i(
            n=-1,
            where_i = [("gt", curr_i)],
            # TODO: tail = from first, asecnding?
        )

        while len(events):

            for e in events:
                i_path = e.obj
                triggers = self.depends.get(i_path)
                if triggers is None:
                    continue
                for o_path, dep in triggers.items():
                    if not dep.active:
                        continue
                    elif dep.events is None:
                        pass
                    elif e.type not in dep.events:
                        continue
                    
                    obj = self.objects.get(o_path)
                    assert obj is not None, o_path

                    handlers = obj.handlers(e, self)
                    for handler in handlers:
                        handler.handle(e, obj, self)

                curr = e
                assert curr.i is not None, curr
                curr_i = curr.i
            
            events = q.head_with_i(
                n=-1,
                where_i = [("gt", curr_i)],
                # TODO: tail = from first, asecnding?
            ) # type: ignore

        if snap is None:
            # TODO: init
            pass
        else:
            snap = replace(snap, event=curr)
            snap.set(self.root.snap_key_val(self))
        
        return self

@dataclass(frozen=True)
class DomainSync(Event):
    obj: str # = domain root
    i: int | None = Event.i_field()
    type: Literal["DomainSync"] = "DomainSync"

@dataclass(frozen=True)
class ObjectNew(Event):
    # NOTE: can't parse old if schema change, if store both
    # we dont store old, so need the diff itself?
    i: int | None = Event.i_field()
    type: Literal["ObjectNew"] = "ObjectNew"

@dataclass(frozen=True)
class ObjectChange(Event):
    # NOTE: can't parse old if schema change, if store both
    # we dont store old, so need the diff itself?
    diff: ObjectDiff
    i: int | None = Event.i_field()
    type: Literal["ObjectChange"] = "ObjectChange"

@dataclass(frozen=True)
class ValueUpdate(Event):
    i: int | None = Event.i_field()
    type: Literal["ValueUpdate"] = "ValueUpdate"

@dataclass(frozen=True)
class ValueAppend(ValueUpdate):
    type: Literal["ValueAppend"] = "ValueAppend"

@dataclass(frozen=True)
class ValuePrepend(ValueUpdate):
    type: Literal["ValuePrepend"] = "ValuePrepend"

@dataclass(frozen=True)
class DefaultEventParser(EventParser):
    event: (
        DomainSync
        | ObjectNew
        | ObjectChange
        | ValueUpdate
        | ValueAppend
        | ValuePrepend
    )
    
__all__ = [
    "Handler",
    "FieldDiff",
    "ObjectDiff",
    "ObjectParser",
    "Object",
    "ObjectKeyVal",
    "ObjectKeyValSQLite",
    #
    "Domain",
    "DomainSync",
    "ObjectNew",
    "ObjectChange",
    "ValueUpdate",
    "ValueAppend",
    "ValuePrepend",
    "DefaultEventParser",
]