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
    Concatenate,Annotated,
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
    Tag,
    Discriminator,
    Field,
)
from frozendict import frozendict

from ..apis import logs
from ..apis import t, kv, assert_is

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .refs import *
from .refs import Ref, IdRef, AliasRef, MemberRef

import os

TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")

T = TypeVar("T")


NULL_ID = ""

def log_when(d: dict):
    when = d.get("when", None)
    if when is None:
        return True
    return bool(os.environ.get(f"LOG.objects.{when}"))


logger, log = logs.logger(
    __file__, streams=dict(formatter=True), when=log_when
)


@dataclass(frozen=True)
class ObjectTagged:
    type: str
    obj: Any


@dataclass(frozen=True)
class ObjectParser():
    obj: Any
    # tagged: Union[
    #     Annotated[ObjectA.Tagged, Tag("ObjectA")],
    #     Annotated[ObjectB.Tagged, Tag("ObjectB")],
    #     # Annotated[None, Tag("None")]
    # ]
    
    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)

TSnap = TypeVar("TSnap", bound=Snap)


# class BaseInterface:

#     # NOTE: subclass as "Interface" if one class per file
#     # or, if many, arg specs can be shared in named interfaces
    
#     # in both cases, do not export

#     class Args:
#         pass

#     class Object:
#         id: str

# def init_new(
#     f: Callable[P, Any],
#     t: Type[O],
# ) -> Callable[P, O]:
#     # TODO: cache fields, encoder
#     def new(*args: P.args, **kwargs: P.kwargs) -> O:
#         parent = args[0]
#         # TODO: type check using parent type from object
#         id = kwargs.pop("uid", None)
#         t_fields = fields(t)
#         i = 0
#         for i, (f, arg) in enumerate(zip(t_fields[2:], args[1:])):
#             kwargs[f.name] = arg
#         i += 2
#         extra_kwargs = {}
#         for f in t_fields[i:]:
#             if f.name in kwargs or f.name in {
#                 "id",
#                 "parent",
#                 "xid",
#                 "uid",
#             }:
#                 continue
#             elif f.default is not MISSING:
#                 extra_kwargs[f.name] = f.default
#             elif f.default_factory is not MISSING:
#                 extra_kwargs[f.name] = f.default_factory()
#         if id is None:
#             xid = kwargs.pop("xid", None)
#             if xid is not None:
#                 id = t.build_id(
#                     parent, str(xid)
#                 )
#             else:
#                 encoder = t.encoder()
#                 id = t.build_id(
#                     parent,
#                     *[
#                         encoder.get(k, str)(v)
#                         for k, v in kwargs.items()
#                     ],
#                 )
#         try:
#             return t(id, parent, **kwargs, **extra_kwargs)
#         except Exception as e:
#             raise ValueError(t, id, args, kwargs, extra_kwargs) from e
#     return new

def build_id(
    parent: str | Ref | Object | None,
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
    elif isinstance(parent, Ref):
        assert isinstance(parent, IdRef), parent
        return f"{parent.id}-{s}"
    else:
        return parent.extend_id(s)

S = TypeVar("S", bound=Snap, covariant=True)

class ObjectWithSnap(Protocol[S]):
    
    @property
    def id(self) -> str: ...

    def new_snap(self) -> S: ...

    def snap_key_val(self) -> SnapKeyVal: ...

@overload
def get_snap(obj: ObjectWithSnap[S], t: None = None) -> S: ...

@overload
def get_snap(obj: ObjectWithSnap[S], t: Type[TSnap]) -> TSnap: ...

def get_snap(
    obj: ObjectWithSnap[S], t: Type[TSnap] | None = None
) -> S | TSnap:

    # TODO: change to use constructor method
    # so dont have to redundantly subclass every time
    # unique on id anyway

    try:
        o = obj.snap_key_val().get(obj.id)
        if o is None:
            o = obj.new_snap()
            # o = type(self).Snap.new(self.id)
        if t is None:
            o = assert_is(o, Snap)
            return cast(S, o)
        return assert_is(o, t)
    except:
        raise ValueError(obj)
    

    
@dataclass(frozen=True)
class ObjectParams:
    
    def encode(self) -> str: ...

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
    # xid: str NOTE: future optimisation, store a hash
    id: str
    parent: str | Object # NOTE: mapped to string on with_obj
    params: ObjectParams | None

    @dataclass(frozen=True)
    class Snap(Snap):
        pass

    @dataclass(frozen=True)
    class Params(ObjectParams):
        pass
    
    # -------------

    def extend_id(self, s: str) -> str:
        return f"{self.id}-{s}"

    # def self(self) -> Obj: ...

    def get_parent(self) -> Object:
        return get_parent(self)

    # @classmethod
    # def encode_type(cls, params: Parm) -> str: ...

    # @classmethod
    # def encode_params(cls, params: Parm) -> str: ...

    @classmethod
    def from_id(cls, s: str):
        # not always implemented, only for fixed, non anonymised, paths
        raise ValueError()

    # TODO: parse/ strip prefix / suffix to get specific values (eg. osc params)

    # TODO: standardise the other from_x constructors?

    # from_parent etc.
    # from_spec (?) (ie. building futures sub derivs)

    # -------------

    def dump_bytes(self) -> bytes:
        self = self.objects_to_ids()
        return context.get().object_parser(
            self #type: ignore
        ).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
    ) -> Object:
        try:
            parser: ObjectParser = context.get().object_parser.ta().validate_json(
                data
            )
        except:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode()
            raise ValueError((data, cls))
        return parser.obj

    def objects_to_ids(self):
        return objects_to_ids(self)

    # -------------

    def handle(
        self, event: Event, snap: Snap
    ) -> frozendict[Snap, Event | tuple[Event, ...]] | None: ...

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

    def new_snap(self) -> Snap:
        raise ValueError(self)

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
            TEST_DB, 
            # context.get().object_parser
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

    def depends_graph(self) -> bool:
        return False
    
    def depends_fields(self) -> dict[str, tuple[str, ...]]:
        return {}

    # NOTE: listen to any event for these objects
    def depends(self) -> list[Dependency]:
        if self.depends_graph():
            deps = [
                Dependency(
                    self.id,
                    assert_is(context.get().name),
                    events=("GraphSync",),
                )
            ]
        else:
            deps = []
        obj_hints = obj_type_hints(type(self))
        for key, events in self.depends_fields().items():
            assert key in obj_hints, (obj_hints, self, key)
            for obj_id in to_obj_ids(getattr(self, key)):
                deps.append(Dependency(
                    self.id,
                    obj_id,
                    events=events
                ))
        return deps

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

    def encode_val(self, key: str, val: Object) -> bytes:
        return val.dump_bytes()

    def decode_val(self, key: str, val: bytes) -> Object:
        assert self.parser is not None, self
        return context.get().object_parser.ta().validate_json(val).obj

@dataclass(frozen=True)
class ObjectKeyValRedis(
    ObjectKeyVal,
    kv.RedisKeyVal[int, str, bytes, str, Object],
):

    @classmethod
    def new(
        cls,
        db: str,
        table: str = "Objects",
        on_update: Callable[[str, Object], None]
        | None = lambda k, o: None,
    ) -> ObjectKeyValRedis:
        self = cls(
            config=kv.RedisKeyValConfig(db=db, name=table),
            on_update=on_update,
        )
        return cast(ObjectKeyValRedis, self.prepare())

@dataclass(frozen=True)
class ObjectKeyValSQLite(
    ObjectKeyVal,
    kv.SQLiteKeyVal[int, str, bytes, str, Object],
):

    @classmethod
    def new(
        cls,
        db: str,
        table: str = "Objects",
        on_update: Callable[[str, Object], None]
        | None = lambda k, o: None,
    ) -> ObjectKeyValSQLite:
        self = cls(
            config=kv.SQLiteKeyValConfig(db=db, name=table),
            on_update=on_update,
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

def resolve_ref(self: Graph, ref: Ref) -> Object | None:
    if isinstance(ref, IdRef):
        return self.objects.get(ref.id)
    if isinstance(ref, AliasRef):
        target = self.aliases.get(ref.name)
        if target is None:
            return None
        return self.resolve_ref(target)
    if isinstance(ref, MemberRef):
        parent_obj = self.resolve_ref(ref.parent)
        if parent_obj is None:
            raise ValueError(ref)
        slot_val = getattr(parent_obj, ref.field)
        if isinstance(slot_val, (tuple, list)):
            item = slot_val[ref.slot]
        elif isinstance(slot_val, (dict, frozendict)):
            item = slot_val[ref.slot]
        else:
            raise ValueError(ref, type(slot_val))
        if isinstance(item, Ref):
            return self.resolve_ref(item)
        if isinstance(item, Object):
            return item
        if isinstance(item, str):
            return self.objects.get(item)
        raise ValueError(item)
    raise ValueError(ref)


O = TypeVar("O", bound=Object)


DEPENDS_FLAGS = {"ALL"}


@dataclass(frozen=True)
class Graph:
    name: str | None = None
    objects: t.frozen.dict[str, Object] = t.frozen.dict()
    depends: t.frozen.dict[str, int] = t.frozen.dict()
    triggers: t.frozen.dict[str, t.frozen.dict[str, Dependency]] =  t.frozen.dict()

    # session-local — not persisted, not consumed by sync().
    aliases: t.frozen.dict[str, Ref] = t.frozen.dict()
    tags:    t.frozen.dict[str, frozenset[str]] = t.frozen.dict()
    info:    t.frozen.dict[str, t.frozen.dict] = t.frozen.dict()

    @property
    def id(self):
        return assert_is(self.name)

    @dataclass(frozen=True)
    class event_parser(EventParser):
        event: (
            GraphSync
            | ObjectChange
            | ObjectNew
            | ValueUpdate
            | ValueAppend
            | ValuePrepend
        )

    @property
    def depends_parser(self) -> Type[DependencyParser]:
        return DependencyParser

    @property
    def object_parser(self) -> Type[ObjectParser]: ...

    @property
    def snap_parser(self) -> Type[SnapParser]: ...

    @property
    def value_parser(self) -> Type[ValueParser]: ...

    def resolve_ref(self, ref: Ref) -> Object | None:
        return resolve_ref(self, ref)

    @overload
    def get(self, obj: None, t: Any) -> None: ...

    @overload
    def get(self, obj: str | Object | Ref, t: Type[O]) -> O: ...

    @overload
    def get(self, obj: str | Object | Ref, t: None) -> Object: ...

    def get(
        self,
        obj: str | Object | Ref | None,
        t: Type[O] | None,
    ) -> O | Object | None:
        o = obj
        if obj is None:
            return obj
        elif isinstance(obj, Ref):
            obj = self.resolve_ref(obj)
        elif isinstance(obj, str):
            obj = self.objects.get(obj)
        if t is None:
            assert isinstance(obj, Object), (o, obj)
            return obj
        assert isinstance(obj, t), (o, obj, t)
        return obj

    def with_ref(
        self,
        parent: Object | Ref | str,
        field: str,
        child: Object,
        *,
        index: int | None = None,
        key: str | None = None,
    ) -> tuple[Graph, Object, Object]:
        """Add `child` to `parent.field` (a ref-bearing slot), deriving the
        child's id from parent + field + slot.

        Slot shape inferred from the field's type annotation on the parent
        class:
          - tuple[<Ref types>, ...] → positional; `index` optional (defaults
            to append). The slot string is `str(index)`.
          - frozendict[str, <Ref types>] / t.frozen.dict[str, ...] → named;
            `key` required. The slot string is the key.

        Child's id is set to `f"{parent.id}-{field}-{slot}"` and its `parent`
        is set to the parent's id. The parent's slot field is mutated to
        include an IdRef to the child.

        Returns (graph, IdRef-to-child).
        """
        parent_obj = (
            parent if isinstance(parent, Object)
            else self.get(parent, Object)
        )
        if parent_obj is None:
            raise ValueError(parent)

        shape = _slot_shape(type(parent_obj), field)
        current = getattr(parent_obj, field)

        slot_n = None

        if shape == "tuple":
            assert key is None, (parent_obj, field, "tuple slot does not accept key")
            slot_n = len(current) if index is None else index
            slot_label = str(slot_n)
            child_id = f"{parent_obj.id}-{field}-{slot_label}"
        elif shape == "dict":
            assert index is None, (parent_obj, field, "dict slot does not accept index")
            assert key is not None, (parent_obj, field, child)
            slot_label = key
            child_id = f"{parent_obj.id}-{field}-{slot_label}"
        else:  # "single"
            assert index is None and key is None, (
                parent_obj, field, "single-ref slot does not accept index/key"
            )
            # No slot label — there's at most one child per single-ref field.
            child_id = f"{parent_obj.id}-{field}"

        # NOTE: everything has their own storage by id
        # so if we already have one, dont overwrite
        # but, if you dont want / need the child to have its own id, leave NULL
        if child.id == NULL_ID:
            child = replace(child, id=child_id, parent=parent_obj.id)

        if shape == "tuple":
            assert slot_n is not None, slot_n
            if slot_n >= len(current):
                new_field = current + (IdRef(child_id),)
            else:
                new_field = (
                    tuple(current[:slot_n])
                    + (IdRef(child_id),)
                    + tuple(current[slot_n + 1:])
                )
        elif shape == "dict":
            new_field = current.set(key, IdRef(child_id))
        else:  # "single"
            new_field = IdRef(child_id)

        parent_obj = replace(parent_obj, **{field: new_field})

        # Insert child and updated parent.
        self = self.with_obj(child)
        self = self.with_obj(parent_obj, force=True)

        return self, parent_obj, child

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
        *,
        alias: str | None = None,
        tag: str | tuple[str, ...] = (),
        info: dict | frozendict | None = None,
    ) -> Graph: ...
    @overload
    def with_obj(
        self,
        obj: Object,
        t: Type[G],
        force: bool = False,
        *,
        alias: str | None = None,
        tag: str | tuple[str, ...] = (),
        info: dict | frozendict | None = None,
    ) -> G: ...
    def with_obj(
        self,
        obj: Object,
        t: Type[G] | None = None,
        force: bool = False,
        *,
        alias: str | None = None,
        tag: str | tuple[str, ...] = (),
        info: dict | frozendict | None = None,
    ) -> Graph | G:
        if obj.id == NULL_ID:
            raise ValueError(obj)
        assert obj.id != self.name, dict(
            obj=obj,
            self=self.name,
        )
        # TODO: proper frozendict subtpye -> replace with set rather than dict spread
        if not force and obj.id in self.objects:
            self = self.apply_labels(obj.id, alias, tag, info)
            if t is not None:
                assert isinstance(self, t), (self, t)
            return self
        if isinstance(obj.parent, Object):
            if not obj.parent.id in self.objects:
                self = self.with_obj(
                    obj.parent, force=force
                )
        obj = obj.objects_to_ids()
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
        self = self.apply_labels(obj.id, alias, tag, info)
        if t is not None:
            assert isinstance(self, t), (self, t)
        context.set(self)
        return self

    def apply_labels(
        self,
        obj_id: str,
        alias: str | None,
        tag: str | tuple[str, ...],
        info: dict | frozendict | None,
    ) -> Graph:
        if alias is not None:
            self = replace(self, aliases=self.aliases.set(alias, IdRef(obj_id)))
        if tag:
            self = self.tag_obj(obj_id, tag)
        if info is not None:
            curr = self.info.get(obj_id, frozendict())
            self = replace(self, info=self.info.set(obj_id, frozendict({**curr, **info})))
        return self

    def tag_obj(
        self,
        obj_id: str,
        tag: str | tuple[str, ...],
    ) -> Graph:
        tags = self.tags
        names = (tag,) if isinstance(tag, str) else tuple(tag)
        for name in names:
            curr = tags.get(name, frozenset())
            tags = tags.set(name, curr | frozenset((obj_id,)))
        return replace(self, tags=tags)

    # ---- alias / tag / info post-creation ----

    def with_alias(self, ref: Ref | str, alias: str) -> Graph:
        ref = ref if isinstance(ref, Ref) else IdRef(ref)
        return replace(self, aliases=self.aliases.set(alias, ref))

    def with_tag(self, ref: Ref | str, tag: str | tuple[str, ...]) -> Graph:
        obj = self.get(ref, Object)
        return self.tag_obj(obj.id, tag)

    def with_info(self, ref: Ref | str, info: dict | frozendict) -> Graph:
        obj = self.get(ref, Object)
        curr = self.info.get(obj.id, frozendict())
        return replace(self, info=self.info.set(obj.id, frozendict({**curr, **info})))

    def get_alias(self, alias: str, t: Type[O] | None = None) -> Object | O:
        ref = self.aliases.get(alias)
        if ref is None:
            raise KeyError(alias)
        return self.get(ref, t)

    # ---- query ----

    def where(
        self,
        *predicates: Callable[[Object], bool],
        paths: dict[str, Any] | None = None,
        type: Type[Object] | tuple[Type[Object], ...] | None = None,
        tag: str | tuple[str, ...] | None = None,
        alias: str | tuple[str, ...] | None = None,
    ):
        """Iterate Objects matching all of the given filters.

        - predicates: callable(obj) -> bool
        - paths: dict[dot-path-string, expected-value] — e.g. {"tags.symbol": "SPY"}
        - type: class or tuple of classes (isinstance check)
        - tag: session tag or tuple of tags (any-of)
        - alias: alias name or tuple of names (any-of)
        """
        # Restrict candidate set if alias/tag given (fast paths).
        if alias is not None:
            names = (alias,) if isinstance(alias, str) else alias
            ids_l = []
            for name in names:
                ref = self.aliases.get(name)
                if ref is None:
                    continue
                obj = self.get(ref, Object)
                if obj is not None:
                    ids_l.append(obj.id)
            candidates = (self.objects[i] for i in ids_l if i in self.objects)
        elif tag is not None:
            tags = (tag,) if isinstance(tag, str) else tag
            ids: set[str] = set()
            for t in tags:
                ids |= self.tags.get(t, frozenset())
            candidates = (self.objects[i] for i in ids if i in self.objects)
        else:
            candidates = self.objects.values()

        for obj in candidates:
            if type is not None and not isinstance(obj, type):
                continue
            if paths is not None and not all(
                _attr_path(obj, p) == v for p, v in paths.items()
            ):
                continue
            if predicates and not all(p(obj) for p in predicates):
                continue
            yield obj

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


@dataclass(frozen=True)
class ObjectChange(Event):
    # NOTE: can't parse old if schema change, if store both
    # we dont store old, so need the diff itself?
    diff: ObjectDiff | None = None


@dataclass(frozen=True)
class ObjectNew(ObjectChange):
    diff: ObjectDiff | None = None


@dataclass(frozen=True)
class ValueUpdate(Event):

    @classmethod
    def if_snap_changed(
        cls,
        old: Snap | None,
        new: Snap,
        into: (
            frozendict[Snap, Event | tuple[Event, ...]] | None
        ) = None,
    ):
        if old is None or old != new:
            if old is not None and old.obj != new.obj:
                raise ValueError((old, new))
            e = ValueUpdate(new.obj)
            if into is None:
                into = frozendict()
            return into.set(new, e)
        return None

    # @overload
    # @classmethod
    # def from_snap(
    #     cls,
    #     snap: Snap,
    #     into: None,
    # ) -> ValueUpdate: ...
    # @overload
    # @classmethod
    # def from_snap(
    #     cls,
    #     snap: Snap,
    #     into: frozendict[str, tuple[Snap, Event]],
    # ) -> frozendict[str, tuple[Snap, Event]]: ...
    # @classmethod
    # def from_snap(
    #     cls,
    #     snap: Snap,
    #     into: (
    #         frozendict[str, tuple[Snap, Event]] | None
    #     ) = None,
    # ):
    #     if into is None:
    #         return ValueUpdate(snap.obj)
    #     return into.set(
    #         snap.obj, (snap, ValueUpdate(snap.obj))
    #     )


@dataclass(frozen=True)
class ValueAppend(ValueUpdate):
    # type: Literal["ValueAppend"] = "ValueAppend"
    pass


@dataclass(frozen=True)
class ValuePrepend(ValueUpdate):
    # type: Literal["ValuePrepend"] = "ValuePrepend"
    pass



@dataclass(frozen=True)
class DependsGraphSync(Object):
    def depends_graph(self):
        return True


@dataclass(frozen=True)
class DependsParent(Object):
    parent: str | Object

    def get_parent(self) -> Object: ...

    def depends_fields(self):
        return {
            "parent": ("ValueUpdate",)
        }



def _attr_path(obj: Any, path: str) -> Any:
    """Walk a dot-separated attribute path on `obj`."""
    cur = obj
    for seg in path.split("."):
        cur = getattr(cur, seg, None)
        if cur is None:
            return None
    return cur


def _is_ref_type(hint: Any) -> bool:
    """True if `hint` is a Ref subclass, or a Union/Optional of Ref subclasses.

    Handles both `Union[A, B]` (typing) and `A | B` (PEP 604) forms.
    """
    if isinstance(hint, type) and issubclass(hint, Ref):
        return True
    import types as _types
    origin = get_origin(hint)
    if origin is Union or origin is _types.UnionType:
        non_none = [a for a in get_args(hint) if a is not type(None)]
        return len(non_none) > 0 and all(_is_ref_type(a) for a in non_none)
    return False


@lru_cache(maxsize=1024 * 4)
def _slot_shape(cls: Type[Any], field: str) -> Literal["tuple", "dict", "single"]:
    """Inspect a dataclass field's type annotation to determine if it is:
      - tuple/list of refs ('tuple') — positional, append/index addressable
      - dict-like keyed by str of refs ('dict') — named, key-addressable
      - a single ref (or Optional thereof) ('single') — replace-only, no slot

    Raises if the field doesn't exist or isn't a recognised slot type.
    """
    hints = get_type_hints(cls)
    hint = hints.get(field)
    if hint is None:
        raise ValueError(
            f"no field {field!r} on {cls.__name__}"
        )
    origin = get_origin(hint)
    if origin in (tuple, list):
        return "tuple"
    if origin is dict:
        return "dict"
    if origin is not None:
        # frozendict / t.frozen.dict / Mapping subclasses
        from collections.abc import Mapping
        if isinstance(origin, type) and (
            issubclass(origin, dict)
            or issubclass(origin, Mapping)
            or "frozendict" in origin.__name__.lower()
        ):
            return "dict"
    if _is_ref_type(hint):
        return "single"
    raise TypeError(
        f"field {field!r} on {cls.__name__} is not a ref-bearing slot "
        f"(needs IdRef, tuple[Ref,...] or frozendict[str,Ref], got {hint!r})"
    )


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


def objects_to_ids(obj: Object):
    hints, obj_hints = obj_type_hints(type(obj))
    assert "parent" in obj_hints, (obj, hints, obj_hints)
    aliases = {
        k: obj_to_alias(getattr(obj, k))
        for k, _ in obj_hints.items()
    }
    return replace(obj, **aliases)

def to_obj_ids(v):
    if v is None:
        return []
    elif isinstance(v, str):
        return [v]
    elif isinstance(v, Object):
        return [v.id]
    elif isinstance(v, IdRef):
        return [v.id]
    elif isinstance(v, (list, tuple)):
        return sum(map(to_obj_ids, v), [])
    elif isinstance(v, (dict)):
        return sum(map(to_obj_ids, v.values()), [])
    else:
        return []
    
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

Obj = TypeVar("Obj", bound=Object, covariant=True)
Par = TypeVar("Par", bound=Object, covariant=True)

TPar = TypeVar("TPar", bound=Object)

Parm = TypeVar("Parm", bound = ObjectParams)
InitP = ParamSpec("InitP")

class ObjectWithParent(Protocol[Par]):

    @property
    def parent(self) -> str | Par: ...

class ObjectProtocol(Protocol[Obj, Par, Parm]):

    @property
    def parent(self) -> str | Par: ...

    @property
    def params(self) -> Parm: ...

    def self(self) -> Obj: ...

    def get_parent(self) -> Par: ...

    @classmethod
    def encode_type(cls, params: Parm) -> str: ...

    @classmethod
    def encode_params(cls, params: Parm) -> str: ...

@overload
def get_parent(obj: ObjectWithParent[Par], t: None = None) -> Par: ...

@overload
def get_parent(obj: ObjectWithParent[Par], t: Type[TPar]) -> TPar: ...

def get_parent(
    obj: ObjectWithParent[Par], t: Type[TPar] | None = None
) -> Par | TPar:
    par = context.get().get(obj.parent, Object)
    if t is None:
        return cast(Par, par)
    return assert_is(par, t)

def encode_type(
    cls: Type[ObjectProtocol[Obj, Par, Parm]],
    params: Parm,
):
    return cls.encode_type(params)

def encode_params(
    cls: Type[ObjectProtocol[Obj, Par, Parm]],
    params: Parm,
):
    return cls.encode_params(params)


def bind(
    cls: Type[ObjectProtocol[Obj, Par, Parm]],
    init_params: Callable[InitP, Parm],
    anon: bool
) -> Callable[
    Concatenate[
        str | Ref | Par,
        str,
        int | str | None,
        InitP
    ], tuple[Par, Obj]]:

    # NOTE: examples of anon:
    # strategy, bind returns: returns depend on strategy
    
    # NOTE: example non anon:
    # strategy, bind signal: strategy depends on signal

    # TODO: need to be able ot pass different parent if non anon?
    # as presumably then the parent is not the node bound to
    # in which case, just use with_ref?

    def init(
        parent: str | Ref | Par,
        field: str,
        index_or_key: int | str | None,
        *args: InitP.args,
        **kwargs: InitP.kwargs
    ) -> tuple[Par, Obj]:
        index: int | None = None
        key: str | None = None
        if isinstance(index_or_key, int):
            index = index_or_key
        elif isinstance(index_or_key, str):
            key = index_or_key
        params = init_params(*args, **kwargs)
        id = NULL_ID if anon else build_id(
            parent,
            cls.encode_type(params),
            cls.encode_params(params),
        )
        obj = cls(id, parent, params) # type: ignore
        obj = cast(Obj, obj)
        _, parent, obj = context.get().with_ref(
            parent, # type: ignore
            field,
            obj,
            index=index,
            key=key,
        )
        return cast(Par, parent), cast(Obj, obj)
    return init

def new(
    cls: Type[ObjectProtocol[Obj, Par, Parm]],
    init_params: Callable[InitP, Parm],
) -> Callable[Concatenate[str | Ref | Par, InitP], Obj]:
    def init(
        parent: str | Ref | Par,
        *args: InitP.args,
        **kwargs: InitP.kwargs
    ) -> Obj:
        params = init_params(*args, **kwargs)
        id = build_id(
            parent,
            cls.encode_type(params),
            cls.encode_params(params),
        )
        obj = cls(id, parent, params) # type: ignore
        return cast(Obj, obj) 
    return init


if TYPE_CHECKING:

    @dataclass(frozen=True)
    class ExampleParams(ObjectParams):
        a: int
        b: float = 1.

    @dataclass(frozen=True)
    class ExampleObj(Object):
        parent: str | Object
        params: ExampleParams

        def self(self) -> ExampleObj:
            return self

        def get_parent(self) -> Object: ...

        @classmethod
        def encode_type(cls, params: ExampleParams) -> str: ...

        @classmethod
        def encode_params(cls, params: ExampleParams) -> str: ...

    example_obj = new(ExampleObj, ExampleParams)
    example_obj("", 1, 1.)

    bind_example_obj = bind(ExampleObj, ExampleParams, anon=False)

__all__ = [
    "P",
    "O",
    
    "FieldDiff",
    "ObjectDiff",
    "ObjectParser",
    "Object",
    "ObjectKeyVal",
    "ObjectKeyValSQLite",
    "ObjectKeyValRedis",
    "objects_to_ids",
    #
    "Graph",
    "G",
    #
    "ObjectNew",
    "ObjectChange",
    "ValueUpdate",
    "ValueAppend",
    "ValuePrepend",
    #
    "GraphSync",
    "DependsGraphSync",
    "DependsParent",
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
    #
    "ObjectProtocol",
    "get_parent",
    "encode_type",
    "encode_params",
    "new",
    "bind",

    "get_snap",
]
