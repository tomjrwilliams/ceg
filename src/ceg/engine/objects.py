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
import heapq
from heapq import heapify, heappush, heappushpop, heappop
import logging
import functools
from functools import partial, lru_cache

import datetime as dt
from datetime import tzinfo

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass
from dataclasses import field, replace, fields, MISSING

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
    __file__, streams=dict(formatter=True), when=log_when
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
    return dataclass(frozen=True)(
        type(
            cls.__name__ + "Parser",
            (ObjectParser,),
            {
                "obj": Field(discriminator="type"),
                "__annotations__": {"obj": cls},
            },
        )
    )


TSnap = TypeVar("TSnap", bound=Snap)


class BaseInterface:

    # NOTE: subclass as "Interface" if one class per file
    # or, if many, arg specs can be shared in named interfaces
    
    # in both cases, do not export

    class Args:
        pass

    class Object:
        id: str

def init_new(
    f: Callable[P, Any],
    t: Type[O],
) -> Callable[P, O]:
    # TODO: cache fields, encoder
    def new(*args: P.args, **kwargs: P.kwargs) -> O:
        parent = args[0]
        # TODO: type check using parent type from object
        id = kwargs.pop("uid", None)
        t_fields = fields(t)
        i = 0
        for i, (f, arg) in enumerate(zip(t_fields[2:], args[1:])):
            kwargs[f.name] = arg
        i += 2
        extra_kwargs = {}
        for f in t_fields[i:]:
            if f.name in kwargs or f.name in {
                "id",
                "parent",
                "xid",
                "uid",
            }:
                continue
            elif f.default is not MISSING:
                extra_kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:
                extra_kwargs[f.name] = f.default_factory()
        if id is None:
            xid = kwargs.pop("xid", None)
            if xid is not None:
                id = t.build_id(
                    parent, str(xid)
                )
            else:
                encoder = t.encoder()
                id = t.build_id(
                    parent,
                    *[
                        encoder.get(k, str)(v)
                        for k, v in kwargs.items()
                    ],
                )
        try:
            return t(id, parent, **kwargs, **extra_kwargs)
        except Exception as e:
            raise ValueError(t, id, args, kwargs, extra_kwargs) from e
    return new

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
    ObjectA(id='a')
    >>> Object.parse_json(b.dump_bytes(), ObjectParserAB)
    ObjectB(id='b', parent='a')
    """

    id: str
    parent: str | Object | None

    # NOTE: string to avoid duplication on insert

    @dataclass(frozen=True)
    class Snap(Snap):
        type: Literal["NullSnap"] = "NullSnap"

    @classmethod
    def encoder(cls) -> frozendict[int | str, Callable[[Any], str]]:
        return frozendict()

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
        repr: bool = False,
    ):
        return Field(default=default, repr=repr)

    @classmethod
    def parser(cls) -> Type[ObjectParser]:
        return object_parser(cls)

    def dump_bytes(self) -> bytes:
        parser = self.parser()
        if isinstance(self.parent, Object):
            self = replace(self, parent=self.parent.id)
        return parser(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[ObjectParser],
    ) -> Object:
        try:
            parser: ObjectParser = into.ta().validate_json(
                data
            )
        except:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode()
            raise ValueError((data, cls))
        return parser.obj

    def check(self):
        assert self.type == type(self).__name__, (
            self, type(self)
        )
        return self

    # def log_repr(self):
    #     return self.alias_parents()

    def alias_parents(self):
        return alias_parents(self)

    def extend_id(self, s: str) -> str:
        return f"{self.id}-{s}"

    @classmethod
    def build_id(
        cls,
        parent: str | Object | None,
        *segs: Any,
        strict: bool = False,
    ):
        s = "-".join([
            s for s in map(str, segs)
            if len(s)
        ])
        if parent is None:
            if not strict:
                return s
            raise ValueError(
                dict(parent=parent, strict=strict)
            )
        elif isinstance(parent, str):
            return f"{parent}-{s}"
        else:
            return parent.extend_id(s)

    # -------------

    @classmethod
    def from_id(cls, s: str):
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
        return kv.get(self.id)

    def set_obj(self) -> Object:
        kv = self.obj_key_val()
        if kv is None:
            raise ValueError(self)
        return kv.set(self.id, self)

    # TODO: separate out the n events
    # from the events themselves

    # and then below we call get event individually?

    def get_event(self, i: int | None = None) -> Event | None:
        kv = self.event_key_val()
        if kv is None:
            raise ValueError(self)
        if i is None:
            res = kv.get(self.id)
            return res
        try:
            return assert_is(kv.get(f"{self.id}/{i}"), Event)
        except Exception as e:
            raise ValueError(self, i) from e

    def extend_events(
        self, events: Sequence[Event]
    ) -> list[Event]:
        kv = self.event_key_val()
        if kv is None:
            raise ValueError(self)
        curr = self.get_event()
        if curr is None:
            acc = 0
        else:
            acc = curr.i + 1
        res = []
        for i, e in enumerate(events):
            i=acc + i
            e = replace(
                e,
                i=i,
                t = dt.datetime.now(dt.UTC).timestamp(),
            )
            e = kv.set(f"{self.id}/{i}", e)
            res.append(e)
            _ = kv.set(self.id, e)
        return res

    # NOTE: not a classmethod, so can be overriden with more specific type by child
    @overload
    def get_snap(self, t: None = None) -> Snap: ...
    @overload
    def get_snap(self, t: Type[TSnap]) -> TSnap: ...
    def get_snap(
        self, t: Type[TSnap] | None = None
    ) -> Snap:
        try:
            o = self.snap_key_val().get(self.id)
            if o is None:
                o = type(self).Snap.new(self.id)
            if t is None:
                return assert_is(o, Snap)
            return assert_is(o, t)
        except:
            raise ValueError(self)

    def _get_snap(self):
        return self.snap_key_val().get(self.id)

    def set_snap(self, snap: Snap) -> Snap:
        return self.snap_key_val().set(self.id, snap)

    # NOTE: not classmethod so can be override more specifically
    def get_value(self) -> Value | None:
        return self.value_key_val().get(self.id)

    def set_value(self, value: Value) -> Value:
        return self.value_key_val().set(self.id, value)

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
    def value_key_val(self, t: Type[T]) -> T: ...

    @overload
    def value_key_val(
        self,
        t: None = None,
    ) -> ValueKeyVal: ...

    def value_key_val(
        self,
        t: Type[T] | None = None,
    ) -> ValueKeyVal | T: ...

    # NOTE: listen to any event for these objects
    def depends(self) -> list[Dependency]: ...

    # NOTE: return the handlers to run for a given event
    # versus having to init all up front, even if not used
    def handler(
        self, event: Event, snap: Snap
    ) -> Handler: ...

    # -------------


# TODO: possibly suffix with asbytes
@dataclass(frozen=True)
class ObjectKeyVal(kv.KeyValStrKeyIOBytesValI[int, Object]):
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[bytes] = bytes
    type_key_o: Type[str] = str
    type_val_o: Type[Object] = Object

    parser: Type[ObjectParser] | None = None

    def encode_val_type(
        self, t: Type[Object]
    ) -> Type[bytes]:
        return bytes

    def encode_val(self, k: str, v: Object) -> bytes:
        return v.dump_bytes()

    def decode_val(self, k: str, v: bytes) -> Object:
        assert self.parser is not None, self
        return Object.parse_json(v, into=self.parser)

@dataclass(frozen=True)
class ObjectKeyValRedis(
    ObjectKeyVal,
    kv.RedisKeyVal[int, str, bytes, str, Object],
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
        cls,
        db: str,
        parser: Type[ObjectParser],
        table: str = "Objects",
        on_update: Callable[[str, Object], None]
        | None = lambda k, o: None,
    ) -> ObjectKeyValRedis:
        self = cls(
            config=kv.RedisKeyValConfig(db=db, name=table),
            on_update=on_update,
            parser=parser,
        )
        return cast(ObjectKeyValRedis, self.prepare())

@dataclass(frozen=True)
class ObjectKeyValSQLite(
    ObjectKeyVal,
    kv.SQLiteKeyVal[int, str, bytes, str, Object],
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
        cls,
        db: str,
        parser: Type[ObjectParser],
        table: str = "Objects",
        on_update: Callable[[str, Object], None]
        | None = lambda k, o: None,
    ) -> ObjectKeyValSQLite:
        self = cls(
            config=kv.SQLiteKeyValConfig(db=db, name=table),
            on_update=on_update,
            parser=parser,
        )
        return cast(ObjectKeyValSQLite, self.prepare())


@dataclass(frozen=True)
class FieldDiff:
    key: str
    old: Any | None
    new: Any | None
    present: tuple[bool, bool]  # old, new


@functools.lru_cache(maxsize=512)
def _fields(t):
    return fields(t)


def diff_field(fld, old, new, flds):
    k = fld.name
    has_old = hasattr(old, k)
    has_new = hasattr(new, k)
    v_old = getattr(old, k, None)
    v_new = getattr(new, k, None)
    if v_old is None and v_new is None:
        return
    if (has_old or has_new) and (v_old != v_new):
        flds[k] = FieldDiff(
            k, v_old, v_new, (has_old, has_new)
        )


@dataclass(frozen=True)
class ObjectDiff:
    id: str
    types: tuple[str, str]
    fields: dict[str, FieldDiff]

    @classmethod
    def new(
        cls,
        old: Object,
        new: Object,
    ) -> ObjectDiff | None:
        flds = {}
        _diff_field = functools.partial(
            diff_field,
            old=old,
            new=new,
            flds=flds,
        )
        _ = list(
            map(
                _diff_field,
                set(_fields(type(old))).union(
                    _fields(type(new))
                ),
            )
        )
        if type(old).__name__ == type(
            new
        ).__name__ and not len(flds):
            return None
        return ObjectDiff(
            new.id,
            (
                type(old).__name__,
                type(new).__name__,
            ),
            flds,
        )


O = TypeVar("O", bound=Object)


DEPENDS_FLAGS = {"ALL"}


@dataclass(frozen=True)
class Graph:
    name: str | None = None
    objects: t.frozen.dict[str, Object] = t.frozen.dict()
    depends: t.frozen.dict[str, int] = t.frozen.dict()
    triggers: t.frozen.dict[str, t.frozen.dict[str, Dependency]] =  t.frozen.dict()

    @property
    def id(self):
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
        t: None = None,
        force: bool = False,
    ) -> Graph: ...
    @overload
    def with_objs(
        self,
        *obj: Object | None,
        t: Type[G],
        force: bool = False,
    ) -> G: ...
    def with_objs(
        self,
        *obj: Object | None,
        t: Type[G] | None = None,
        force: bool = False,
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
        self,
        obj: Object,
        t: None = None,
        force: bool = False,
    ) -> Graph: ...
    @overload
    def with_obj(
        self,
        obj: Object,
        t: Type[G],
        force: bool = False,
    ) -> G: ...
    def with_obj(
        self,
        obj: Object,
        t: Type[G] | None = None,
        force: bool = False,
    ) -> Graph | G:
        assert obj.id != self.name, dict(
            obj=obj,
            self=self.name,
        )
        # TODO: proper frozendict subtpye -> replace with set rather than dict spread
        if not force and obj.id in self.objects:
            if t is not None:
                assert isinstance(self, t), (self, t)
            return self
        if isinstance(obj.parent, Object):
            if not obj.parent.id in self.objects:
                self = self.with_obj(
                    obj.parent, force=force
                )
        obj = obj.check().alias_parents()
        objects = objects = self.objects.set(obj.id, obj)
        
        triggers = self.triggers
        obj_depends = obj.depends()

        self = replace(
            self, depends = self.depends.set(
                obj.id, 0 if obj_depends is None else len(obj_depends)
            )
        )

        # TODO: we key on object, be very careful that deps are specified at once per trigger

        if obj_depends is not None:
            for dep in obj_depends:
                assert dep.obj == obj.id
                obj_peers = triggers.get(dep.depends, frozendict())
                triggers = triggers.set(
                    dep.depends, obj_peers.set(obj.id, dep)
                )
        self = replace(
            self, objects=objects, triggers=triggers
        )
        if t is not None:
            assert isinstance(self, t), (self, t)
        context.set(self)
        return self

    # NOTE: we dont need to diff the domain itself
    # if we add objects then their depends will be picked up on sync, and we know to iter the whole queue again, regardless of the domains own latest
    def sync(
        self,
        n: int | None = None,
        **log_kwargs
    ) -> Graph:
        log("INFO")(sync=self.name, **log_kwargs)

        snaps: frozendict[str, Snap] = frozendict()
        events: frozendict[str, Event] = (
            frozendict()
        )

        for k, obj_new in self.objects.items():
            obj_new, events = sync_object_def(
                events,
                obj_new,
                k,
            )

        name = assert_is(self.name)

        snaps = snaps.set(name, Snap(name))
        events = events.set(name, GraphSync(name, i=0, t= dt.datetime.now(
            dt.UTC
        ).timestamp()))

        queued_graph_sync = False
        queue = []

        heapify(queue)

        # TODO: redis pubsub if using?
        # else - is there a way to have theevents written by sub proc as well?

        queued: frozendict[str, set[int]] = frozendict()

        for obj_path, obj in self.objects.items():
            snap = obj.get_snap(Snap)
            snaps = snaps.set(obj_path, snap)

            depends = obj.depends() or []
            
            for dep in depends:
                parent_path = dep.depends
                if (
                    dep.events != ("GraphSync",)
                    and parent_path not in self.objects
                ):
                    log("WARNING")(parent_not_found=parent_path, for_obj=obj_path)
                    continue

                i = snap.handled.get(parent_path, -1)

                parent_latest = events.get(parent_path, None)
                if parent_latest is None:
                    parent = self.get(parent_path, Object)
                    parent_latest = parent.get_event()
                    events = events.set(parent_path,parent_latest)

                parent_latest = assert_is(parent_latest, Event)

                parent_queued = queued.get(parent_path, set())

                if i < parent_latest.i or isinstance(
                    parent_latest,
                    GraphSync
                ):
                    if isinstance(parent_latest, GraphSync):
                        e = parent_latest
                        if not queued_graph_sync:
                            heappush(
                                queue,
                                (e.t, e)
                            )
                            queued_graph_sync = True
                    else:
                        i += 1
                        if i in parent_queued:
                            continue
                        parent = self.get(parent_path, Object)
                        e = parent.get_event(i)
                        parent_queued = parent_queued | {i}
                        heappush(
                            queue,
                            (e.t, e)
                        )

                queued = queued.set(parent_path, parent_queued)

        i = 0
        while len(queue):
            t, e = heappop(queue)
            e = assert_is(e, Event)

            if isinstance(e, GraphSync):
                trigger = self
            elif e.obj not in self.objects:
                # NOTE: created event for not bound object
                continue
            else:
                trigger = self.get(e.obj, Object)


            deps = self.triggers.get(e.obj, {})

            for dep_path, dep in deps.items():
                
                dep_obj = self.get(dep_path, Object)
                dep_snap = snaps[dep_path]

                dep_handled = dep_snap.handled.get(e.obj, -1)

                if e.i is None:
                    raise ValueError(e)

                if dep_handled < e.i:

                    snaps, events = handle_event(
                        dep_obj,
                        trigger,
                        dep_snap,
                        e,
                        dep,
                        queue,
                        snaps,
                        events,
                        # changes,
                        self.depends,
                    )

                if isinstance(trigger, Object):
                    next_i = e.i + 1
                    if next_i <= events[e.obj].i and (
                        next_i not in queued[e.obj]
                    ):
                        next_e = trigger.get_event(next_i)
                        heappush(queue, (next_e.t, next_e))

            i += 1
            if n is not None and i >= n:
                break

        return self

def handle_event(
    obj: Object,
    trigger: Object  | Graph,
    snap: Snap,
    e: Event,
    dep: Dependency,
    queue: list,
    snaps: frozendict[str, Snap],
    events: frozendict[str, Event],
    # changes: frozendict[str, int],
    n_depends: frozendict[str, int],
):
    e_type_match = (
        dep.events is None
        or e.type in dep.events
    )

    if e_type_match:
        h = obj.handler(e, snap)
    else:
        h = None

    # log("INFO")(e=e, h=h)

    fire_triage = False
    set_latest = False

    if snap.triage:
        set_latest = h is not None
        if len(snap.latest) < n_depends[obj.id] - 1:
            h = None
        elif (
            len(snap.latest) == n_depends[obj.id] - 1
            and snap.latest.get(e.obj, 0.) == 0.
        ):
            fire_triage=True
        else:
            raise ValueError(snap)
        
    run = h is not None

    if snap.triage:
        log("INFO")(e=e, obj=obj.id, run = (run, set_latest))
    elif run:
        log("INFO")(e=e, obj=obj.id, run = run, h = type(h))
    else:
        log("INFO")(e=e, obj=obj.id, run = run)

    if not run:
        # e.type not in dep.events
        # log("INFO")(
        #     when="skip_event",
        #     trigger=(
        #         f"{type(trigger).__name__}"
        #         f"({trigger.id})"
        #         f".{e.type}"
        #         f".{e.i}"
        #     ),
        #     skipping=(f"{type(obj).__name__}({obj.id})"),
        # )
        pass
    else:
        # log("INFO")(
        #     when="handle_event",
        #     trigger=(
        #         f"{type(trigger).__name__}"
        #         f"({trigger.id})"
        #         f".{e.type}"
        #         f".{e.i}"
        #     ),
        #     handling=(
        #         f"{type(obj).__name__}({obj.id})"
        #         f".{type(h).__name__}"
        #     ),
        # )
        caused: frozendict[
            str, tuple[Snap, list[Event]]
        ] = h.handle(e, snap, obj)

        snaps, events = update_snaps(
            obj,
            trigger,
            e,
            caused,
            snaps,
            events,
            # changes,
            queue,
        )

    snap = assert_is(snaps.get(snap.obj), Snap)

    snap = (
        snap
        if isinstance(trigger, Graph)
        else snap.set_handled(
            trigger.id,
            (assert_is(e.i, int), assert_is(e.t, float)),
        )
    )
    if set_latest:
        snap = snap.set_latest(trigger.id, (
            e.i, e.t
        ) if not fire_triage else None)

    snaps = snaps.set(snap.obj, obj.set_snap(snap))
    return snaps, events

# TODO: in theory, handle can be farmed out to sub process?
# just have to change the model so that we get objects from store
# eg. redis
# rather than central dict

# TODO: if objects register who they update snaps on behalf of
# can make sure we never dispatch to two updating the same
# then we can batch out to sub process for each round of event deps

# TODO:
# when we call update snap in the handle method we check it's alreayd been registered

# TODO: also then helps with the issue of tracing dependency graph
# as previously those edges weren't available

# TODO: so we can then do an up front cyclicality check on graph sync
# and broken edge check - no unreachable nodes

# TODO: so we have to pass down those dpeendent snaps as well?
# to the sub process
# in theory they can do their own update calls saving events, latest etc
# and then we return them back from the subprocess

def update_snaps(
    obj: Object,
    trigger: Object | Graph,
    e: Event,
    caused: frozendict[str, tuple[Snap, list[Event]]],
    snaps: frozendict[str, Snap],
    events: frozendict[str, Event],
    # changes: frozendict[str, int],
    queue,
):
    self: Graph = context.get(Graph)
    for snap_path, (snap, es) in caused.items():

        assert snap_path != trigger.id, dict(
            snap=snap,
            trigger=trigger.id
        )

        if isinstance(es, Event):
            es = [es]
        try:
            snap_obj: Object = self.get(snap_path, Object)
        except:
            raise ValueError(
                (
                    snap_path,
                    type(obj),
                )
            )
        if len(es):
            es = snap_obj.extend_events(es)
            events = events.set(
                snap_path, es[-1]
            )
            for e in es:
                heappush(queue, (e.t, e))

        snaps = snaps.set(
            snap_path, snap_obj.set_snap(snap)
        )

    return snaps, events


G = TypeVar("G", bound=Graph)
P = ParamSpec("P")


def sync_object_def(
    events: frozendict[str, Event],
    obj: Object,
    k: str,
):
    obj_old = obj.get_obj()

    if obj_old is None:
        log("INFO")(when="new_object", obj_new=k)
        e = ObjectNew(k, i=0)
        es = obj.extend_events([e])
        events = events.set(
            obj.id, es[-1]
        )
        obj.set_obj()
    else:
        diff = obj_old.diff(obj)
        if diff is None:
            # TODO: listen to object sync
            # method on object if they publish sync?

            return obj, events
        log("INFO")(when="diff_object", object_changed=diff)
        es = obj.extend_events([ObjectChange(k, diff)])
        events = events.set(
            obj.id,
            es[-1]
        )
        obj.set_obj()
    return obj, events

@dataclass(frozen=True)
class GraphSync(Event):
    obj: str  # = graph name
    i: int | None = Event.i_field()
    t: float | None = Event.t_field()
    type: Literal["GraphSync"] = "GraphSync"


@dataclass(frozen=True)
class ObjectChange(Event):
    # NOTE: can't parse old if schema change, if store both
    # we dont store old, so need the diff itself?
    diff: ObjectDiff | None
    i: int | None = Event.i_field()
    t: float | None = Event.t_field()
    type: Literal["ObjectChange"] = "ObjectChange"


@dataclass(frozen=True)
class ObjectNew(ObjectChange):
    diff: ObjectDiff | None = None
    type: Literal["ObjectNew"] = "ObjectNew"


@dataclass(frozen=True)
class ValueUpdate(Event):
    i: int | None = Event.i_field()
    t: float | None = Event.t_field()
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
        into: frozendict[str, tuple[Snap, Event]],
    ) -> frozendict[str, tuple[Snap, Event]]: ...
    @classmethod
    def from_snap(
        cls,
        snap: Snap,
        into: (
            frozendict[str, tuple[Snap, Event]] | None
        ) = None,
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
                self.id,
                assert_is(context.get().name),
                events=("GraphSync",),
            )
        ]

    def depends(self):
        return self.depends_graph()


@dataclass(frozen=True)
class DependsParamsValue(Object):
    params: t.frozen.dict[str, str]
    
    def depends(self) -> list[Dependency]:
        g = context.get()
        parent = g.get(self.parent, Object)
        env = os.environ.get("ENV", "prod")
        try:
            return [
                # Dependency(
                #     self.id,
                #     assert_is(context.get().name),
                #     events=("ValueUpdate",),
                # )
            ] + [
                Dependency(
                    self.id,
                    p if env == "test" else g.get(p, Object).id,
                    events=("ValueUpdate",),
                )
                for p in (
                    self.params
                    if isinstance(self.params, tuple)
                    else self.params.values()
                )
            ]
        except:
            raise ValueError(self)


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
                self.id,
                parent.id,
                events=("ObjectNew",),
            )
        ]

    # TODO: change, value, etc.


@dataclass(frozen=True)
class DependsParentValue(DependsParent):
    # TODO: object_update?
    def handler_object_new(self) -> Handler: ...

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            assert self.get_parent().id == event.obj, (
                self,
                event,
            )
            return self.handler_object_new()

        raise ValueError((self, event))

    def depends(self) -> list[Dependency]:
        return self.depends_parent()

    def depends_parent(self) -> list[Dependency]:
        return [
            Dependency(
                self.id,
                assert_is(
                    self.parent
                    if isinstance(self.parent, str)
                    else self.parent
                    if self.parent is None
                    else self.parent.id
                ),
                events=("ValueUpdate",),
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
                self.id,
                assert_is(grand).id,
                events=("ValueUpdate",),
                # TODO: append, prepend?
            )
        ]


@dataclass(frozen=True)
class DependsSelfNew(DependsSelf):
    def depends(self) -> list[Dependency]:
        return [
            Dependency(
                self.id, self.id, events=("ObjectNew",)
            )
        ]


@dataclass(frozen=True)
class DependsSelfNewUpdate(DependsSelf):
    def depends(self) -> list[Dependency]:
        return [
            Dependency(
                self.id,
                self.id,
                events=(
                    "ObjectNew",
                    "ValueUpdate",
                ),
            ),
        ]


@dataclass(frozen=True)
class DependsSelfUpdate(DependsSelf):
    def depends(self) -> list[Dependency]:

        # TODO: how to not fire continually?

        return [
            Dependency(
                self.id,
                self.id,
                events=("ValueUpdate",),
            )
        ]


def is_obj_class(t: Type[Any]):
    return t is Object or issubclass(t, Object)


def is_obj_hint(hint: Any):
    if isinstance(hint, type) and is_obj_class(hint):
        return True
    elif get_origin(hint) is Union:
        return any([is_obj_hint(h) for h in get_args(hint)])
    else:
        return False


@lru_cache(maxsize=512)
def obj_type_hints(t: Type[Object]):
    hints = get_type_hints(t)
    return hints, {
        k: hint
        for k, hint in hints.items()
        # if is_obj_hint(hint)
    }


def obj_to_alias(v: Any):
    # TODO: wrap in type v if subclass of container?
    if isinstance(v, Object):
        return v.id
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
    obj: tuple[Object | str, ...],
) -> tuple[str, ...]: ...


def obj_path(obj: Object | str | tuple[Object | str, ...]):
    if isinstance(obj, tuple):
        return tuple([obj_path(o) for o in obj])
    return obj if isinstance(obj, str) else obj.id


def find_object(
    obj: Object | str, objs: tuple[Object | str, ...] | None
) -> int | None:
    if objs is None:
        return None
    id=obj_path(obj)
    for i, o in enumerate(objs):
        if obj_path(o) == id:
            return i
    return None


# TODO; context update for multi proc
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
        into: Type[UpdateParser],
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
    return dataclass(frozen=True)(
        type(
            cls.__name__ + "Parser",
            (UpdateParser,),
            {
                "snap": Field(discriminator="type"),
                "__annotations__": {"snap": cls},
            },
        )
    )


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
            kwargs=kwargs,
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

    def drop(self, name: str):
        self.graphs = self.graphs.delete(name)

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
        target: str | None = None,  # if none, current
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
    "BaseInterface",
    "init_new",
    "P",
    "O",
    "Handler",
    "FieldDiff",
    "ObjectDiff",
    "ObjectParser",
    "Object",
    "ObjectKeyVal",
    "ObjectKeyValSQLite",
    "ObjectKeyValRedis",
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
    "DependsParamsValue",
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
