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
    Sequence,
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
from dataclasses import field, replace, fields

from contextlib import contextmanager

import multiprocessing as mp

from pydantic import (
    TypeAdapter,
    BaseModel,
    Field,
)
from frozendict import frozendict

from ..apis import logs
from ..apis import t, kv, assert_is

from .events import *
from .snaps import *
from .values import *
from .depends import *

import os

TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

T = TypeVar("T")

def log_when(d: dict):
    when = d.get("when", None)
    if when is None:
        return True
    return bool(os.environ.get(f"LOG.objects.{when}"))

logger, log = logs.logger(
    __file__,
    streams = dict(formatter=True),
    when=log_when
)

@dataclass(frozen=True)
class Handler:
    
    def handle(
        self,
        event: Event,
        snap: Snap,
        obj: Object,
    ) -> frozendict[str, tuple[Snap, list[Event]]]: ...

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

TSnap = TypeVar("TSnap", bound=Snap)

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
    
    @dataclass(frozen=True)
    class Snap(Snap):
        type: Literal["NullSnap"] = "NullSnap"

    def get_parent(self):
        return context.get().get(self.parent, Object)

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
        try:
            parser: ObjectParser = (
                into.ta().validate_json(data)
            )
        except:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode()
            raise ValueError((data, cls))
        return parser.obj

    def check(self):
        assert self.type == type(self).__name__, self
        return self
    
    # def log_repr(self):
    #     return self.alias_parents()

    def alias_parents(self):
        return alias_parents(self)

    def extend_path(self, s: str) -> str:
        return f"{self.path}-{s}"
    
    @classmethod
    def build_path(
        cls,
        parent: str | Object | None,
        *segs: Any,
        strict: bool = False,
    ):
        s = "-".join(map(str, segs))
        if parent is None:
            if not strict:
                return s
            raise ValueError(dict(
                parent=parent, strict=strict
            ))
        elif isinstance(parent, str):
            return f"{parent}-{s}"
        else:
            return parent.extend_path(s)
        

    # -------------
    
    @classmethod
    def from_path(cls, s: str):
        # not always implemented, only for fixed, non anonymised, paths
        raise ValueError()
    
    # TODO: parse/ strip prefix / suffix to get specific values (eg. osc params)

    # TODO: standardise the other from_x constructors?

    # from_parent etc.
    # from_spec (?) (ie. building futures sub derivs)

    # -------------
    
    def diff(self, new: Object) -> ObjectDiff | None:
        return ObjectDiff.new(self, new)

    # TODO: separate classmethods that arent override for getting obj, snap, value, etc.

    # NOTE: not classmethod so can be override more specifically
    def get_obj(self) -> Object | None: 
        kv = self.obj_key_val()
        if kv is None:
            raise ValueError(self)
        return kv.get(self.path)

    def set_obj(self) -> Object:
        kv = self.obj_key_val()
        if kv is None:
            raise ValueError(self)
        return kv.set(self.path, self)

    def get_events(
        self
    ) -> EventList:
        kv = self.event_key_val()
        if kv is None:
            raise ValueError(self)
        res = kv.get(self.path)
        if res is None:
            return EventList()
        return res
    
    def extend_events(
        self, events: Sequence[Event]
    ) -> list[Event]:
        kv = self.event_key_val()
        if kv is None:
            raise ValueError(self)
        acc = self.get_events()
        return kv.set(
            self.path,
            acc + [
                replace(e, i=len(acc) + i)
                for i, e in enumerate(events)
            ]
        )

    # NOTE: not a classmethod, so can be overriden with more specific type by child
    @overload
    def get_snap(self, t: None = None) -> Snap: ...
    @overload
    def get_snap(self, t: Type[TSnap]) -> TSnap: ...
    def get_snap(self, t: Type[TSnap] | None =None) -> Snap:
        try:
            o = self.snap_key_val().get(self.path)
            if o is None:
                o = type(self).Snap.new(self.path)
            if t is None:
                return assert_is(o, Snap)
            return assert_is(o, t)
        except:
            raise ValueError(self)
        
    def _get_snap(self):
        return self.snap_key_val().get(self.path)

    def set_snap(self, snap: Snap) -> Snap:
        return self.snap_key_val().set(self.path, snap)

    # NOTE: not classmethod so can be override more specifically
    def get_value(self) -> Value | None:
        return self.value_key_val().get(self.path)

    def set_value(self, value: Value) -> Value:
        return self.value_key_val().set(self.path, value)
        
    # NOTE: value change isnt emit by set by default
    # rather, up to object to decide if updating their own value, to emit the event (as part of the handle)
    # eg. for append / prepend - more specific than just update

    # -------------
    # NOTE: roughly in order of likelihood of override

    def event_key_val(self) -> EventKeyVal: ...

    # NOTE: also generally not likely often overriden
    def dependency_key_val(self) -> DependencyKeyVal: ...

    def obj_key_val(self) -> ObjectKeyVal:
        return ObjectKeyValSQLite.new(
            TEST_DB, context.get().object_parser
        )

    # TODO: generic one implemented at root that has a switch on fields on object for backend type and params (eg. type=sqlite, db=..., table=...)
    # and io as SnapRawIO (bytes i, bytes o)
    # but more likely local IO overriden
    def snap_key_val(self) -> SnapKeyVal: ...

    # NOTE: as for snap, but likely more object specific, migth be key_val subtype (eg. chunked), depending on value type (get_value then calls value_key_val)
    @overload
    def value_key_val(
        self, t: Type[T]
    ) -> T: ...

    @overload
    def value_key_val(
        self, t: None = None,
    ) -> ValueKeyVal: ...

    def value_key_val(
        self, 
        t: Type[T] | None = None,
    ) -> ValueKeyVal | T: ...

    # NOTE: listen to any event for these objects
    def depends(self) -> list[Dependency]: ...

    # NOTE: return the handlers to run for a given event
    # versus having to init all up front, even if not used
    def handler(self, event: Event, snap: Snap) -> Handler: ...

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
        on_update: Callable[[str, Object], None] | None = lambda k, o: None
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

@functools.lru_cache(maxsize=512)
def _fields(t):
    return fields(t)

def diff_field(
    fld, old, new, flds
):
    k = fld.name
    has_old = hasattr(old, k)
    has_new = hasattr(new, k)
    v_old = getattr(old, k, None)
    v_new = getattr(new, k, None)
    if v_old is None and v_new is None:
        return
    if (has_old or has_new) and (v_old != v_new):
        flds[k] = FieldDiff(
            k,
            v_old,
            v_new,
            (has_old, has_new)
        )

@dataclass(frozen=True)
class ObjectDiff:
    path: str
    types: tuple[str, str]
    fields: dict[str, FieldDiff]

    @classmethod
    def new(
        cls, old: Object, new: Object,
    ) -> ObjectDiff | None:
        flds = {}
        _diff_field = functools.partial(
            diff_field,
            old=old,
            new=new,
            flds=flds,
        )
        _ = list(map(_diff_field, set(
            _fields(type(old))
        ).union(
            _fields(type(new))
        )))
        if (
            type(old).__name__ == type(new).__name__
            and not len(flds)
        ):
            return None
        return ObjectDiff(
            new.path,
            (
                type(old).__name__,
                type(new).__name__,
            ),
            flds
        )

O = TypeVar("O", bound=Object)

@dataclass(frozen=True)
class Graph:
    name: str | None = None
    objects: dict[str, Object] = Field(
        default_factory=dict
    )
    triggers: dict[str, dict[str, Dependency]] = Field(
        default_factory=dict
    )

    @property
    def path(self):
        # TODO: replace name with path?
        return assert_is(self.name)
    
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
    def get(self, obj: None, t: Any) -> None: ...

    @overload
    def get(self, obj: str | Object, t: Type[O]) -> O: ...

    @overload
    def get(self, obj: str | Object, t: None) -> Object: ...

    def get(
        self, 
        obj: str | Object | None,
        t: Type[O] | None,
    ) -> O | Object | None:
        o = obj
        if obj is None:
            return obj
        elif isinstance(obj, str):
            obj = self.objects.get(obj)
        if t is None:
            assert isinstance(obj, Object), (o, obj)
            return obj
        assert isinstance(obj, t), (o, obj, t)
        return obj
    
    # TODO: use update?
    # what other changes actually are there?

    # just set and update obj?
    # (for update encoding?)

    @overload
    def with_objs(
        self,
        *obj: Object | None,
        t: None=None,
        force: bool = False
    ) -> Graph: ...
    @overload
    def with_objs(
        self,
        *obj: Object | None,
        t: Type[G],
        force: bool = False
    ) -> G: ...
    def with_objs(
        self,
        *obj: Object | None,
        t: Type[G] | None = None,
        force: bool = False
    ) -> Graph | G:
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
    ) -> Graph: ...
    @overload
    def with_obj(
        self, obj: Object, t: Type[G], force: bool = False, 
    ) -> G: ...
    def with_obj(
        self, obj: Object, t: Type[G] | None = None, force: bool = False
    ) -> Graph | G:
        assert obj.path != self.name, dict(
            obj=obj,
            self=self.name,
        )
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
        triggers = self.triggers
        obj_depends = obj.depends()

        # TODO: we key on object, be very careful that deps are specified at once per trigger

        if obj_depends is not None:
            for dep in obj_depends:
                assert dep.obj == obj.path
                obj_peers = triggers.get(dep.depends, {})
                triggers = {
                    **triggers,
                    dep.depends: {
                        **obj_peers,
                        obj.path: dep
                    }
                }
        self = replace(
            self, objects=objects, triggers=triggers
        )
        if t is not None:
            assert isinstance(self, t), (self, t)
        context.set(self)
        return self
    
    # NOTE: we dont need to diff the domain itself
    # if we add objects then their depends will be picked up on sync, and we know to iter the whole queue again, regardless of the domains own latest
    def sync(self) -> Graph:
        log("INFO")( sync=self.name)

        snaps: frozendict[str, Snap] = frozendict()
        events: frozendict[
            str, tuple[Event, ...]
        ] = frozendict()
        
        for k, obj_new in self.objects.items():
            obj_new = sync_object_def(
                events,
                obj_new,
                k,
            )

        name = assert_is(self.name)

        snaps = snaps.set(name, Snap(name))
        events = events.set(name, (GraphSync(
            name, i = 0
        ),))

        # TODO: anything that has changed def (not value)
        # propagate
        # drop event history
        # drop all values
        # re-fire created

        # separate to the normal flow?

        # TODO: can then use the version flag, if you load from args for any to auto increment
        # to force all downstream of that node
        # eg. if they're project children

        # ie. a 'domain' is just a root node with a verison flag you can use to easily flush all the non fixed path objects

        # rename project

        triggered = set(self.triggers.keys())

        while len(triggered):
            changed: dict[str, int] = {}
            for trigger in triggered:
                obj_deps = self.triggers.get(trigger, {})

                trigger_obj = (
                    self
                    if trigger == name
                    else (
                        context.get()
                        .get(trigger, Object)
                    )
                )

                (
                    snaps, events, obj_changes
                ) = handle_events(
                    trigger_obj,
                    obj_deps,
                    snaps=snaps,
                    events=events,
                )
                changed = {
                    **changed,
                    **obj_changes,
                }
            triggered = set([
                k for k, n in changed.items() if n > 0
            ])
        
        return self


G = TypeVar("G", bound=Graph)
P = ParamSpec("P")

def sync_object_def(
    events: frozendict[str, tuple[Event, ...]],
    obj: Object,
    k: str,
):
    obj_old = obj.get_obj()
    
    if obj_old is None:
        log("INFO")(when="new_object", obj_new=k)
        e = ObjectNew(k, i = 0)
        events = events.set(
            obj.path,
            tuple(obj.extend_events([e]))
        )
        obj.set_obj()
    else:
        diff = obj_old.diff(obj)
        if diff is None:
            
            # TODO: listen to object sync
            # method on object if they publish sync?

            return obj
        log("INFO")(when="diff_object", object_changed=diff)
        events = events.set(
            obj.path,
            tuple(obj.extend_events([
                ObjectChange(k, diff)
            ]))
        )
        obj.set_obj()
    return obj

def handle_events(
    trigger: Object | Graph,
    depends: dict[str, Dependency],
    snaps: frozendict[str, Snap],
    events: frozendict[str, tuple[Event, ...]],
):
    changes: frozendict[str, int] = frozendict()

    if trigger.path not in snaps:
        snap = assert_is(
            trigger, Object
        ).get_snap(Snap)
        snaps = snaps.set(trigger.path, snap)
    else:
        snap = snaps[trigger.path]

    if trigger.path not in events:
        trigger_events = assert_is(
            trigger, Object
        ).get_events()
        events = events.set(
            trigger.path, tuple(trigger_events)
        )
    else:
        trigger_events = events[trigger.path]

    for k, dep in depends.items():
        obj = context.get().get(k, Object)
        if obj.path not in snaps:
            obj_snap = obj.get_snap(Snap)
            snaps = snaps.set(obj.path, obj_snap)
        else:
            obj_snap = snaps[obj.path]

        trigger_handled = obj_snap.handled.get(
            trigger.path, -1
        )
        for e in sorted(
            trigger_events,
            key = lambda e: assert_is(e.i),
        ):
            if trigger_handled < assert_is(e.i):
                if (
                    dep.events is None
                    or e.type in dep.events
                ):
                    h = obj.handler(e, obj_snap)
                else:
                    h = None
                    
                (
                    snaps, 
                    events,
                    changes,
                ) = handle_event(
                    obj,
                    trigger,
                    obj_snap,
                    e,
                    h,
                    snaps,
                    events,
                    changes
                )

    # TODO: if has batching on obj
    # acc the triggering events
    # and dispatch at once (eg. latest_value_update?)
    
    return snaps, events, changes

def handle_event(
    obj: Object,
    trigger: Object | Graph,
    snap: Snap,
    e: Event,
    h: Handler | None,
    snaps: frozendict[str, Snap],
    events: frozendict[str, tuple[Event, ...]],
    changes: frozendict[str, int],
):
    if h is None:
        # e.type not in dep.events
        log("INFO")(
            when="skip_event",
            trigger=(
                f"{type(trigger).__name__}"
                f"({trigger.path})"
                f".{e.type}"
                f".{e.i}"
            ),
            skipping=(
                f"{type(obj).__name__}({obj.path})"
            )
        )
        caused = frozendict({
            # snap.obj: (snap, [])
        })
    else:
        log("INFO")(
            when="handle_event",
            trigger=(
                f"{type(trigger).__name__}"
                f"({trigger.path})"
                f".{e.type}"
                f".{e.i}"
            ),
            handling=(
                f"{type(obj).__name__}({obj.path})"
                f".{type(h).__name__}"
            )
        )
        caused: frozendict[
            str, tuple[Snap, list[Event]]
        ] = h.handle(e, snap, obj)

    if caused is None:
        raise ValueError((obj, h))

    snaps, events, changes = update_snap(
        obj,
        trigger,
        e,
        caused,
        snaps,
        events,
        changes,
    )

    # TODO: for collections, this gets quite slow
    # better to update just the one field handled rather than rewrite the whole dict
    # ie. have to store each handled entry as a separate row / key?

    snap = assert_is(snaps.get(snap.obj), Snap)
    snap = replace(
        snap, 
        handled=(
            snap.handled
            if isinstance(trigger, Graph)
            else snap.handled.set(
                trigger.path, assert_is(e.i),
            )
        )
    )
    snaps = snaps.set(
        snap.obj, obj.set_snap(snap)
    )
    return snaps, events, changes

def update_snap(
    obj: Object,
    trigger: Object | Graph,
    e: Event,
    caused: frozendict[
        str, tuple[Snap, list[Event]]
    ],
    snaps: frozendict[str, Snap],
    events: frozendict[str, tuple[Event, ...]],
    changes: frozendict[str, int],
):
    self = context.get()
    for snap_path, (snap, es) in caused.items():
        if isinstance(es, Event):
            es = [es]
        try:
            snap_obj = self.get(snap_path, Object)
        except:
            raise ValueError((
                snap_path,
                type(obj),
            ))
        
        events = events.set(
            snap_path, 
            tuple(snap_obj.extend_events(es))
        )
        snaps = snaps.set(
            snap_path, snap_obj.set_snap(snap)
        )
        changes = changes.set(
            snap_path,
            changes.get(snap_path, 0) + len(es)
        )
    
    return snaps, events, changes

@dataclass(frozen=True)
class GraphSync(Event):
    obj: str # = graph name
    i: int | None = Event.i_field()
    type: Literal["GraphSync"] = "GraphSync"

@dataclass(frozen=True)
class ObjectChange(Event):
    # NOTE: can't parse old if schema change, if store both
    # we dont store old, so need the diff itself?
    diff: ObjectDiff | None
    i: int | None = Event.i_field()
    type: Literal["ObjectChange"] = "ObjectChange"

@dataclass(frozen=True)
class ObjectNew(ObjectChange):
    diff: ObjectDiff | None = None
    type: Literal["ObjectNew"] = "ObjectNew"


@dataclass(frozen=True)
class ValueUpdate(Event):
    i: int | None = Event.i_field()
    type: Literal["ValueUpdate"] = "ValueUpdate"

    @overload
    @classmethod
    def from_snap(
        cls,
        snap: Snap,
        into: None,
    ) -> ValueUpdate: ...
    @overload
    @classmethod
    def from_snap(
        cls,
        snap: Snap,
        into: frozendict[str, tuple[Snap, Event]]
    ) -> frozendict[str, tuple[Snap, Event]]: ...
    @classmethod
    def from_snap(
        cls,
        snap: Snap,
        into: (
            frozendict[str, tuple[Snap, Event]]
            | None
        ) = None
    ):
        if into is None:
            return ValueUpdate(snap.obj)
        return into.set(
            snap.obj, (snap, ValueUpdate(snap.obj))
        )
        

@dataclass(frozen=True)
class ValueAppend(ValueUpdate):
    type: Literal["ValueAppend"] = "ValueAppend"

@dataclass(frozen=True)
class ValuePrepend(ValueUpdate):
    type: Literal["ValuePrepend"] = "ValuePrepend"

@dataclass(frozen=True)
class DefaultEventParser(EventParser):
    event: (
        GraphSync
        | ObjectChange
        | ObjectNew
        | ValueUpdate
        | ValueAppend
        | ValuePrepend
    )
    
@dataclass(frozen=True)
class DependsSelf(Object):
    def depends_self(self) -> list[Dependency]: ...

@dataclass(frozen=True)
class DependsGraphSync(Object):
    def depends_graph(self) -> list[Dependency]:
        return [
            Dependency(
                self.path,
                assert_is(context.get().name),
                events=("GraphSync",)
            )
        ]

    def depends(self):
        return self.depends_graph()
    
@dataclass(frozen=True)
class DependsParent(Object):
    parent: str | Object

    def get_parent(self) -> Object: ...

@dataclass(frozen=True)
class DependsParentNew(DependsParent):
    def depends(self) -> list[Dependency]:
        parent = context.get().get(self.parent, Object)
        return [
            Dependency(
                self.path,
                parent.path,
                events=("ObjectNew",)
            )
        ]
    
    # TODO: change, value, etc.

@dataclass(frozen=True)
class DependsParentValue(DependsParent):

    # TODO: object_update?
    def handler_object_new(self) -> Handler: ...

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            assert (
                self.get_parent().path == event.obj
            ), (self, event)
            return self.handler_object_new()

        raise ValueError((self, event))
    
    def depends(self) -> list[Dependency]:
        return self.depends_parent()
    
    def depends_parent(self) -> list[Dependency]:
        return [
            Dependency(
                self.path,
                assert_is(
                    self.parent
                    if isinstance(self.parent, str)
                    else self.parent
                    if self.parent is None
                    else self.parent.path
                ),
                events=("ValueUpdate",)
                # TODO: append, prepend?
            )
        ]
@dataclass(frozen=True)
class DependsGrandParentValue(DependsParent):
    def depends(self) -> list[Dependency]:
        return self.depends_grand_parent()
    
    def depends_grand_parent(self) -> list[Dependency]:
        parent = context.get().get(self.parent, Object)
        grand = assert_is(
            context.get().get(parent.parent, Object)
        )
        return [
            Dependency(
                self.path,
                assert_is(grand).path,
                events=("ValueUpdate",)
                # TODO: append, prepend?
            )
        ]
    

@dataclass(frozen=True)
class DependsSelfNew(DependsSelf):
        
    def depends(self) -> list[Dependency]:
        return [
            Dependency(
                self.path,
                self.path,
                events=(
                    "ObjectNew",
                )
            )
        ]
    
@dataclass(frozen=True)
class DependsSelfNewUpdate(DependsSelf):
        
    def depends(self) -> list[Dependency]:
        return [
            Dependency(
                self.path,
                self.path,
                events=(
                    "ObjectNew",
                    "ValueUpdate",
                )
            ),
        ]
    
@dataclass(frozen=True)
class DependsSelfUpdate(DependsSelf):
        
    def depends(self) -> list[Dependency]:

        # TODO: how to not fire continually?
        
        return [
            Dependency(
                self.path,
                self.path,
                events=("ValueUpdate",)
            )
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

@overload
def obj_path(
    obj: Object | str,
) -> str: ...

@overload
def obj_path(
    obj: tuple[Object | str, ...]
) -> tuple[str,  ...]: ...

def obj_path(
    obj: Object | str | tuple[Object | str, ...]
):
    if isinstance(obj, tuple):
        return tuple([obj_path(o) for o in obj])
    return obj if isinstance(obj, str) else obj.path

def find_object(
    obj: Object | str, objs: tuple[Object | str, ...] | None
) -> int | None:
    if objs is None:
        return None
    path = obj_path(obj)
    for i, o in enumerate(objs):
        if obj_path(o) == path:
            return i
    return None


@dataclass(frozen=True)
class Update:

    # TODO: i: int (for order in sync)
    
    @property
    def type(self) -> str: ...

    @classmethod
    def union_field(cls):
        return Field(discriminator="type")

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[UpdateParser]:
        return update_parser(cls)

    def dump_bytes(self) -> bytes:
        return self.parser()(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[UpdateParser]
    ) -> Update:
        parser: UpdateParser = into.ta().validate_json(data)
        return parser.update

@dataclass(frozen=True)
class UpdateParser:
    update: Update

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)

@functools.lru_cache(maxsize=128, typed=True)
def update_parser(cls: Type[Update]) -> Type[UpdateParser]:
    return dataclass(frozen=True)(type(
        cls.__name__ + "Parser",
        (UpdateParser,),
        {
            "snap": Field(discriminator="type"),
            "__annotations__": {"snap": cls}
        }
    ))

@dataclass(frozen=False)
class EncodedContext:
    queue: mp.Queue
    context: bytes

    # TODO: parser??

    def spawn(
        self,
        group: None = None,
        target: Callable | None = None,
        name: str | None = None, 
        daemon: bool | None = None,
        *args,
        **kwargs,
    ):
        return mp.Process(
            group=group,
            target=target,
            name=name,
            daemon=daemon,
            args=(self.queue, self.context) + args,
            kwargs=kwargs
        )


@dataclass(frozen=False)
class Context:
    # TODO: last update int

    # TODO: parser?

    # TODO: generalise to generic?
    stack: list[str]
    # TODO: frozen
    graphs: t.frozen.dict[str, Graph] = frozendict()

    safe: bool = True
    # queue: mp.Queue | None = None

    def set(self, graph: Graph):
        if not self.safe:
            # NOTE: as would require entire encode/decode
            raise ValueError()
        # self.graphs = context.graphs.set(name, graph)
        self.graphs = self.graphs.set(
            assert_is(graph.name), graph
        )

    # def assume_safe(self):
    #     if not self.safe:
    #         raise ValueError()
    #     return self, assert_is(self.queue)

    # def encode(self): # TODO: rename to do with multi
    #     self, queue = self.assume_safe()
    #     b: bytes = b'' # TODO
    #     return EncodedContext(queue, b)
    
    # @classmethod
    # def decode( # TODO: rename to do with multi
    #     cls,
    #     queue: mp.Queue,
    #     b: bytes,
    # ):
    #     # TODO: decode (pydnatic?)
    #     return cls([], queue=queue, safe=True)

    def update(self, update: Update):
        # TODO: apply the update, last updated
        return

    # TODO: various sub methods of graph update spawn
    # if safe, publish 
    # else do directly (?)

    def sync(self):
        # TODO: consume any updates from queue until we are latest or none
        # apply
        return

    @overload
    def get(
        self,
        t: None,
        name: str | None = None,
    ) -> Graph | None: ...

    @overload
    def get(
        self,
        t: Type[G] = Graph,
        name: str | None = None,
    ) -> G: ...

    def get(
        self,
        t: Type[Graph] | None = Graph,
        name: str | None = None,
    ):
        # TODO: if safe, consume all updates?
        # or only on set?
        if name is None and not len(self.stack):
            if t is None:
                return None
            raise ValueError(t)
        name = name or self.stack[-1]
        graph = self.graphs.get(name, None)
        if t is not None:
            assert isinstance(graph, t), (graph, name, t)
        return graph

    @contextmanager
    def graph(
        self,
        name: str,
        t: Type[G] = Graph,
        # other kwargs
    ):
        # TODO: if safe, consume any new changes
        # TODO: then publish update
        # TODO: hen apply update

        # else do the below
        try:
            if name not in self.graphs:
                graph = t(name)
                self.set(graph)
            self.stack.append(name)
            yield
        finally:
            assert name == self.stack.pop(), (name, self)

    @contextmanager
    def fork_graph(
        self,
        name: str,
        target: str | None = None, # if none, current
    ):
        # existing graph, but with new attributes
        # use .graph()
        try:
            raise ValueError()
        finally:
            pass

# def _example():
#     with context.graph("a"):
#         g1 = context.get()
#         print("g1:", g1)
#         with context.graph("b"):
#             g2 = context.get()
#             assert g2 != g1, (g2, g1)
#             print("g2:", g2)
#             with context.graph("a"):
#                 g3 = context.get()
#                 assert g3 != g2, (g3, g2)
#                 print("g3:", g3, "eq:", id(g1) == id(g3))
#             g4 = context.get()
#             print("g2:", g4, "eq:", id(g2) == id(g4))
#         g5 = context.get()
#         print("g1:", g5, "eq:", id(g1) == id(g5))

#     assert context.get(t=None) is None, context

# if TYPE_CHECKING:
#     pass
# else:
#     import pydantic
#     pydantic.dataclasses.rebuild_dataclass(Object)
#     pydantic.dataclasses.rebuild_dataclass(Graph)

context = Context([])

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
    "Graph",
    "G",
    #
    "ObjectNew",
    "ObjectChange",
    "ValueUpdate",
    "ValueAppend",
    "ValuePrepend",
    "DefaultEventParser",
    #
    "DependsSelf",
    "DependsGraphSync",
    "GraphSync",
    "DependsSelfNew",
    "DependsSelfUpdate",
    "DependsGraphSync",
    "DependsParent",
    "DependsParentNew",
    "DependsGrandParentValue",
    "DependsParentValue",
    #
    "obj_path",
    "find_object",
    #
    "Update",
    "UpdateParser",
    "update_parser",
    "EncodedContext",
    "Context",
    "context",
]