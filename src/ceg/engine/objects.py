from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    ParamSpec,
    Any,
    Union,
    overload,
    cast,
    get_type_hints,
    get_args,
    get_origin,
    final,
)
import logging
import functools
from functools import partial, lru_cache

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

from ..apis import logs
from ..apis import kv

from .events import *
from .snaps import *
from .values import *
from .depends import *

import os

TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

T = TypeVar("T")

logger, log_kw = logs.prepare_logger(__file__)

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
    
    def log_repr(self):
        return self.alias_parents()

    def alias_parents(self):
        return alias_parents(self)

    def extend_path(self, s: str) -> str:
        return f"{self.path}-{s}"

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
        try:
            return self.snap_key_val(domain).get(self.path)
        except:
            raise ValueError(self)

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
    @overload
    def value_key_val(
        self, domain: Domain, t: Type[T]
    ) -> T: ...

    @overload
    def value_key_val(
        self, domain: Domain, t: None = None,
    ) -> ValueKeyVal: ...

    def value_key_val(
        self, 
        domain: Domain,
        t: Type[T] | None = None,
    ) -> ValueKeyVal | T: ...

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
        self = cls(config=kv.SQLiteKeyValConfig(
            db=db, name=table
        ), on_update=on_update, parser=parser)
        return cast(ObjectKeyValSQLite, self.prepare())

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

O = TypeVar("O", bound=Object)

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

    @overload
    def get(self, obj: None, t: Type[O]) -> None: ...

    @overload
    def get(self, obj: str | Object, t: Type[O]) -> O: ...

    @overload
    def get(self, obj: str | Object, t: None) -> Object: ...

    def get(
        self, 
        obj: str | Object | None,
        t: Type[O] | None,
    ) -> O | Object | None:
        if obj is None:
            return obj
        elif isinstance(obj, str):
            obj = self.objects.get(obj)
        if t is None:
            assert isinstance(obj, Object), obj
            return obj
        assert isinstance(obj, t), (obj, t)
        return obj
    
    @overload
    def with_objs(
        self,
        *obj: Object | None,
        t: None=None,
        force: bool = False
    ) -> Domain: ...
    @overload
    def with_objs(
        self,
        *obj: Object | None,
        t: Type[D],
        force: bool = False
    ) -> D: ...
    def with_objs(
        self,
        *obj: Object | None,
        t: Type[D] | None = None,
        force: bool = False
    ) -> Domain | D:
        for o in obj:
            if o is None:
                continue
            self = self.with_obj(o, force=force)
        if t is not None:
            assert isinstance(self, t), (self, t)
        return self

    @overload
    def with_obj(
        self, obj: Object, t: None=None, force: bool = False, 
    ) -> Domain: ...
    @overload
    def with_obj(
        self, obj: Object, t: Type[D], force: bool = False, 
    ) -> D: ...
    def with_obj(
        self, obj: Object, t: Type[D] | None = None, force: bool = False
    ) -> Domain | D:
        # TODO: proper frozendict subtpye -> replace with set rather than dict spread
        if not force and obj.path in self.objects:
            if t is not None:
                assert isinstance(self, t), (self, t)
            return self
        if isinstance(obj.parent, Object):
            if not obj.parent.path in self.objects:
                self = self.with_obj(
                    obj.parent, force=force
                )
        obj = obj.check().alias_parents()
        objects={
            **self.objects,
            obj.path: obj
        }
        depends = self.depends
        obj_depends = obj.depends(self)
        if obj_depends is not None:
            for dep in obj_depends:
                assert dep.obj == obj.path
                obj_peers = depends.get(dep.depends, {})
                depends = {
                    **depends,
                    dep.depends: {
                        **obj_peers,
                        obj.path: dep
                    }
                }
        self = replace(
            self, objects=objects, depends=depends
        )
        if t is not None:
            assert isinstance(self, t), (self, t)
        return self
    
    # NOTE: we dont need to diff the domain itself
    # if we add objects then their depends will be picked up on sync, and we know to iter the whole queue again, regardless of the domains own latest
    def sync(self) -> Domain:
        
        log = partial(
            log_kw, prefix=f"sync({self.root.path})"
        )
        log(start=True)
        
        # TODO: or object.value not snap?
        snap = self.root.get_snap(self)
        if snap is None:
            curr = None
            curr_i = -1
        else:
            curr = snap.event
            assert curr.i is not None, curr
            curr_i = curr.i
        
        log(snap=snap, curr=curr)

        q = self.root.event_queue(self)

        events: list[Event]
        history: list[Event] | None = None

        for i_path, triggers in self.depends.items():
            for o_path, dep in triggers.items():
                obj = self.objects.get(o_path)
                assert obj is not None, o_path
                obj, dep = handle_past_events(
                    self,
                    obj=obj,
                    dep=dep,
                    curr_i=curr_i,
                    i_path = i_path,
                    log=log,
                    q=q,
                    history=history,
                )

        e_sync = self.root.publish(DomainSync(
            self.root.path,
        ), self)
        assert e_sync.i is not None, e_sync
        
        for k, obj_new in self.objects.items():
            obj_new = sync_object_def(
                self,
                obj_new,
                k,
                log=log,
            )

        events = q.head_with_i(
            n=-1,
            where_i = [("gt", curr_i)],
            # TODO: tail = from first, asecnding?
        )

        while len(events):
            for e in events:
                curr, curr_i = handle_new_event(
                    self,
                    e,
                    curr_i,
                    log=log,
                )

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

D = TypeVar("D", bound=Domain)
P = ParamSpec("P")

def sync_object_def(
    self: Domain,
    obj_new: Object,
    k: str,
    log: Callable[..., None],
):
    obj_old = obj_new.get_obj(self)

    if obj_old is None:
        log(obj_new=obj_new)
        e_new = self.root.publish(ObjectNew(k), self)
        assert e_new.i is not None, e_new
    else:
        diff = obj_old.diff(obj_new)
        log(obj_diff=diff)
        if diff is None:
            return obj_new
        e_chg = self.root.publish(
            ObjectChange(k, diff), self
        )
        assert e_chg.i is not None, e_chg
    return obj_new

def handle_past_events(
    self: Domain,
    obj: Object,
    dep: Dependency,
    curr_i: int,
    i_path: str,
    log: Callable[..., None],
    q: EventQueue,
    history: list[Event] | None = None,
) -> tuple[Object, Dependency]:

    log(obj=obj, dep=dep)

    if not dep.active:
        dep = dep.set(
            obj.dependency_key_val(self)
        )
        return obj, dep

    dep_prev = dep.get(
        dep.key(),
        obj.dependency_key_val(self)
    )
    log(obj=obj, dep_prev=dep_prev)

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

    log(obj=obj, n_events=len(events))

    for e in events:

        handlers = obj.handlers(e, self)
        log(obj=obj, e=e, n_handlers=len(handlers))

        for handler in handlers:
            log(obj=obj, e=e, handler=handler, start=True)
            handler.handle(e, obj, self)
            log(obj=obj, e=e, handler=handler, end=True)

    dep = dep.set(
        obj.dependency_key_val(self)
    )
    return obj, dep

def handle_new_event(
    self: Domain,
    e: Event,
    curr_i: int,
    log: Callable[..., None],
) -> tuple[Event, int]:
    i_path = e.obj
    triggers = self.depends.get(i_path)

    if triggers is None:
        log(e=e, triggers=triggers)

        assert e.i is not None, e
        return e, e.i

    for o_path, dep in triggers.items():
        if not dep.active:
            continue
        elif dep.events is None:
            pass
        elif e.type not in dep.events:
            continue
            
        obj = self.objects.get(o_path)
        assert obj is not None, o_path

        log(obj=obj, dep=dep, e=e)

        handlers = obj.handlers(e, self)
        log(obj=obj, e=e, handlers=handlers)

        for handler in handlers:
            log(obj=obj, e=e, handler=handler, start=True)
            handler.handle(e, obj, self)
            log(obj=obj, e=e, handler=handler, end=True)

    curr = e
    assert curr.i is not None, curr
    curr_i = curr.i

    return curr, curr_i

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
    
@dataclass(frozen=True)
class DependsSelf(Object):
    def depends_self(self, domain: Domain) -> list[Dependency]: ...

@dataclass(frozen=True)
class DependsDomain(Object):
    def depends_self(self, domain: Domain) -> list[Dependency]: ...
    
@dataclass(frozen=True)
class DependsSelfNew(Object):
        
    def depends_self(self, domain: Domain) -> list[Dependency]:
        # TODO: listen to ancestors?
        
        # TODO: self.ObjectChange
        # for chunk and backend type / loc

        return [
            Dependency(
                self.path,
                self.path,
                events=("ObjectNew",)
            )
        ]
    
@dataclass(frozen=True)
class DependsDomainSync(Object):
    def depends_domain(self, domain: Domain) -> list[Dependency]:
        return [
            Dependency(
                self.path,
                domain.root.path,
                events=("DomainSync",)
            ),
        ]

def is_obj_class(t: Type[Any]):
    return t is Object or issubclass(t, Object)

def is_obj_hint(hint: Any):
    if isinstance(hint, type) and is_obj_class(hint):
        return True
    elif get_origin(hint) is Union:
        return any([
            is_obj_hint(h) for h in get_args(hint)
        ])
    else:
        return False

@lru_cache(maxsize=512)
def obj_type_hints(t: Type[Object]):
    hints = get_type_hints(t)
    return hints, {
        k: hint for k, hint in hints.items()
        # if is_obj_hint(hint)
    }

def obj_to_alias(v: Any):
    # TODO: wrap in type v if subclass of container?
    if isinstance(v, Object):
        return v.path
    elif isinstance(v, list):
        return [obj_to_alias(vv) for vv in v]
    elif isinstance(v, tuple):
        return tuple((obj_to_alias(vv) for vv in v))
    elif isinstance(v, dict):
        return {
            obj_to_alias(k): obj_to_alias(vv)
            for k, vv in v.items()
        }
    elif isinstance(v, set):
        return set((obj_to_alias(vv) for vv in v))
    # elif isinstance(v, frozendict): ...
    else:
        return v

def alias_parents(obj: Object):
    hints, obj_hints = obj_type_hints(type(obj))
    assert "parent" in obj_hints, (obj, hints, obj_hints)
    aliases = {
        k: obj_to_alias(getattr(obj, k))
        for k, _ in obj_hints.items()
    }
    return replace(obj, **aliases)

__all__ = [
    "O",
    "Handler",
    "FieldDiff",
    "ObjectDiff",
    "ObjectParser",
    "Object",
    "ObjectKeyVal",
    "ObjectKeyValSQLite",
    "alias_parents",
    #
    "Domain",
    "DomainSync",
    "ObjectNew",
    "ObjectChange",
    "ValueUpdate",
    "ValueAppend",
    "ValuePrepend",
    "DefaultEventParser",
    #
    "DependsSelf",
    "DependsDomain",
    "DependsSelfNew",
    "DependsDomainSync",
]