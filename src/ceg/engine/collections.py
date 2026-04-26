from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    Concatenate,
    overload,
    cast,
    final,
)
from pathlib import Path
import functools
import itertools
from collections import defaultdict

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

from ..apis import t, kv, assert_is

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *



@dataclass(frozen=True)
class CollectionParser:
    coll: Collection

    @classmethod
    def ta(cls) -> TypeAdapter:
        return TypeAdapter(cls)

    def dump_bytes(self) -> bytes:
        return self.ta().dump_json(self)


@functools.lru_cache(maxsize=128, typed=True)
def collection_parser(cls: Type[Collection]) -> Type[CollectionParser]:
    return dataclass(frozen=True)(
        type(
            cls.__name__ + "Parser",
            (CollectionParser,),
            {
                "coll": Field(discriminator="type"),
                "__annotations__": {"coll": cls},
            },
        )
    )


@dataclass(frozen=True)
class Collection:

    name: str | None = None

    objects: t.frozen.dict[str, Object] = frozendict()
    tags: t.frozen.dict[str, frozenset[str]] = frozendict()
    info: t.frozen.dict[str, frozendict] = frozendict()

    prefix: str | None = None

    type: Literal["Collection"] = "Collection"

    @classmethod
    def type_field(cls, name: str, repr: bool = False):
        return Field(default=name, repr=repr)

    @classmethod
    def parser(cls) -> Type[CollectionParser]:
        return collection_parser(cls)

    def dump_bytes(self) -> bytes:
        parser = self.parser()
        return parser(self).dump_bytes()

    @classmethod
    def parse_json(
        cls,
        data: str | bytes | bytearray,
        into: Type[CollectionParser],
    ) -> Collection:
        try:
            parser: CollectionParser = into.ta().validate_json(
                data
            )
        except:
            if isinstance(data, (bytes, bytearray)):
                data = data.decode()
            raise ValueError((data, cls))
        return parser.coll

    def with_name(self, name: str):
        return replace(self, name=name)
    
    def set_prefix(self, prefix: str | None = None):
        prefix = prefix or self.name
        if prefix is None:
            raise ValueError(self)
        return replace(self, prefix=prefix)

    def iter(self):
        yield from self.objects.items()

    def with_prefix(self, key: str):
        return replace(
            self,
            objects=frozendict({
                f"{key}/{k}": o for k, o in self.objects.items()
            }),
            tags=frozendict({
                t: frozenset([
                    f"{key}/{k}" for k in ks
                ]) for t, ks in self.tags.items()
            }),
            info=frozendict({
                f"{key}/{k}": i for k, i in self.info.items()
            }),
            prefix=None,
        )

    def with_obj(
        self,
        obj: Object,
        key: str,
        tag: str | tuple[str, ...] = (),
        info: frozendict = frozendict(),
    ):
        if not isinstance(key, str):
            raise ValueError(key)
        objects = self.objects
        objects = objects.set(key, obj)
        tags = self.tags
        if isinstance(tag, str):
            tag = (tag,)
        for t in tag:
            acc = ""
            for seg in t.split("/"):
                acc += seg
                keys = tags.get(acc, set())
                tags = tags.set(acc, frozenset(keys | {key}))
        return replace(
            self,
            objects=objects,
            tags=tags,
            info=self.info.set(key, info)
        )
    
    def get(
        self,
        key: str | tuple[str, ...]
    ):
        if isinstance(key, str):
            f = None
            if "." in key and key.endswith("()"):
                key, f = key.rsplit(".", maxsplit=1)
                f = f[:-2]
            v = self.objects.get(key)
            if f is not None:
                return getattr(v, f)()
            return v
        return tuple([
            self.objects.get(k) for k in key
        ])
    def where(
        self,
        key: str | tuple[str, ...] | None = None,
        tag: str | tuple[str, ...] | None = None,
        info: frozendict | tuple[frozendict, ...] | None = None,
        filters: Filter | tuple[Filter, ...] | None = None,
    ):
        objects = self.objects
        if isinstance(key, str):
            obj = objects.get(key, None)
            if obj is None:
                return None
            objects = frozendict({key: obj})
        elif isinstance(key, tuple):
            objects = frozendict({
                k: obj
                for k, obj in (
                    (k, objects.get(k, None))
                    for k in key
                )
                if obj is not None
            })
        if isinstance(tag, str):
            tag = (tag,)
        if isinstance(tag, tuple):
            keys = set()
            for t in tag:
                keys = keys | self.tags.get(key, set())
            objects = frozendict({
                k: obj
                for k, obj in objects.items()
                if k in keys
            })
        elif tag is not None:
            raise ValueError(tag)
        if isinstance(info, (dict, frozendict)):
            info = (info,)
        if isinstance(info, tuple):
            res = {}
            for i in info:
                res = res | {
                    k: obj
                    for k, obj in objects.items()
                    if info_matches(
                        self.info.get(k, frozendict()), i
                    )
                }
            objects = frozendict(res)
        elif info is not None:
            raise ValueError(info)
        if isinstance(filters, Filter):
            filters = filters,
        if isinstance(filters, tuple):
            objects = frozendict({
                k: obj
                for k, obj in objects.items()
                if all([
                    f(obj, self.info.get(k, frozendict()))
                    for f in filters
                ])
            })
        elif filters is not None:
            raise ValueError(filters)
        return objects

    
    @classmethod
    def merge(
        cls,
        *collections: Collection,
        **prefixed_collections: Collection
    ):
        colls =list(collections)
        for k, collection in prefixed_collections.items():
            collection= collection.with_prefix(k)
            colls.append(collection)
        res = None
        keys = None
        for i, collection in enumerate(colls):
            if collection.prefix:
                collection = collection.with_prefix(collection.prefix)
            if res is None:
                res = collection.objects
                keys = set(collection.objects.keys())
            else:
                coll_keys = set(collection.objects.keys())
                overlap = keys.intersection(coll_keys)
                if len(overlap):
                    raise ValueError((overlap, i))
                keys = keys | coll_keys
                res = res | collection.objects
        # TODO: tags, info
        return Collection(objects=res)
    
def info_matches(
    info: frozendict,
    matches: frozendict,
):
    return all ([
        (
            i_v is None and v is None
        ) or v == i_v
        for k, i_v, v
        in (
            (k, info.get(k), v)
            for k, v in matches.items()
        )
    ])

def bound(uid: str):
    return f"o:{assert_is(uid, str)}"

def unpack_bound(
    v, k: str, bound: Bound, coll: Collection, force: bool = False,
):
    orig = v
    if v is None:
        return v
    elif isinstance(v, str):
        if not v.startswith("o:") and not force:
            return v
        f = None
        if "." in v and v.endswith("()"):
            v, f = v.rsplit(".", maxsplit=1)
            f = f[:-2]
        try:
            if force and not v.startswith("o:"):
                v = coll.get(v)
            else:
                v = coll.get(v[2:])
            if f is not None:
                v = getattr(v, f)()
            v = assert_is(v)
        except Exception as e:
            raise ValueError(
                "Can't unpack ref.\nCollection objects:\n"
                + "\n".join(map(str, coll.objects.keys()))
                + f"\norig={orig}\nk={k}\nf={f}\n{bound}"
            ) from e
        return v.id
    elif isinstance(v, tuple):
        return tuple([
            unpack_bound(vv, k, bound, coll, force=force)
            for vv in v
        ])
    elif isinstance(v, list):
        return [
            unpack_bound(vv, k, bound, coll, force=force)
            for vv in v
        ]
    elif isinstance(v, frozendict):
        return frozendict({
            kk: unpack_bound(vv, k, bound, coll, force=force)
            for kk, vv in v.items()
        })
    elif isinstance(v, dict):
        return {
            kk: unpack_bound(vv, k, bound, coll, force=force)
            for kk, vv in v.items()
        }
    elif hasattr(v, "unpack"):
        return v.unpack(bound, coll, force=force)
    else:
        raise ValueError(v, k, bound)

@dataclass(frozen=True)
class Bound:
    func: Callable[P, O]
    name: str
    params: t.frozen.dict = frozendict()
    tag: str | tuple[str, ...] = ()
    info: t.frozen.dict = frozendict()

    container: str | None = None

    unpack_fields: tuple[str, ...] = ()

    def with_info(self, **info):
        return replace(self, info = frozendict({
            **self.info,
            **info,
        }))

    def with_params(self, **params):
        return replace(self, params = frozendict(
            **self.params,
            **params
        ))

    def unpack(self, *fields: str):
        return replace(self, unpack_fields=fields)

    def bind(
        self,
        collection: Collection,
        force: bool = False,
    ):
        name = self.name
        params = {
            k: (
                unpack_bound(v, k, self, collection, force=True)
                if k in self.unpack_fields
                else v if not isinstance(v, str)
                else unpack_bound(v, k, self, collection)
            )
            for k, v in self.params.items()
        }
        if "parent" in params:
            parent = params.pop("parent")
            try:
                obj = self.func(parent, **params)
            except Exception as e:
                raise ValueError(self, params) from e
            context.get().with_obj(obj, force=force)
        elif self.container:
            assert self.container.startswith("o:"), self
            # TODO: force unpack, to make it more straight forwad to bidn
            container = unpack_bound(
                self.container, "container", self, collection
            )
            container, obj = self.func(container, **params)

            container_name = self.container[2:]
            
            collection = replace(
                collection,
                objects=collection.objects.set(container_name, container)
            )
        else:
            raise ValueError(self)
        if name.startswith("o:"):
            raise ValueError(self, name)
        if "\n" in name:
            raise ValueError(self, name)
        return collection.with_obj(
            obj, name, tag=self.tag, info=self.info
        )

from dataclasses import fields

def init_bind(
    func: Callable[P, O],
    t: Type[O],
) -> Callable[
    Concatenate[
        str
        | tuple[str, str | tuple[str, ...]]
        | tuple[str, frozendict]
        | tuple[str, str | tuple[str, ...] | frozendict], 
        P
    ],
    Bound
]:
    def new(
        bind: (
            str
            | tuple[str, str | tuple[str, ...]]
            | tuple[str, frozendict]
            | tuple[str, str | tuple[str, ...] | frozendict]
        ),
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Bound:
        if isinstance(bind, str):
            name = bind
            info = frozendict()
            tag = ()
        elif isinstance(bind, tuple):
            if len(bind) == 2:
                name, tag_or_info = bind
                if isinstance(tag_or_info, frozendict):
                    info = tag_or_info
                    tag = ()
                elif isinstance(tag_or_info, (str, tuple)):
                    tag = tag_or_info
                    info = frozendict()
                else:
                    raise ValueError(bind)
            elif len(bind) == 3:
                name, tag, info = bind
            else:
                raise ValueError(bind)
        else:
            raise ValueError(bind)
        t_fields = fields(t)

        for i, (f, arg) in enumerate(zip(t_fields[1:], args)):
            kwargs[f.name] = arg
        return Bound(
            func,
            name,
            params = frozendict(kwargs),
            tag=tag,
            info=info,
        )
    return new

def bind_params(
    parent: str | Object | Bound,
    **kwargs
):
    if isinstance(parent, Bound):
        parent = f"o:{parent.name}"
    elif isinstance(parent, str) and not parent.startswith("o:"):
        parent = bound(parent)
    return frozendict(dict(parent=parent, **kwargs))

@dataclass(frozen=True)
class Universe:
    params: tuple[t.frozen.dict, ...] = ()

    def add(self, **params):
        return replace(self, params = self.params + (frozendict(params),))
    
    def generate(self, collection: Collection):
        return self.params
    
    def zip(self):
        # TODO: zip with eg. a predicate?
        return self


def unpack_universe_params(ps: tuple[frozendict | dict]):
    p, *ps = ps
    if not len(ps):
        res = tuple([
            frozendict({
                k: v for k, v in zip(p.keys(), vs)
            })
            for vs in zip(*p.values())
        ])
        return res
    return tuple(map(frozendict, (p,) + ps))

def universe(
    *params: frozendict | dict,
    **product: list,
) -> Universe:
    if len(params):
        params = unpack_universe_params(params)
    if len(params) and len(product):
        product_params = tuple([
            frozendict({
                k: v
                for k, v in zip(product.keys(), vs)
            })
            for vs in itertools.product(
                *product.values()
            )
        ])
        return Universe(tuple([
            p | product_p
            for p, product_p
            in itertools.product(params, product_params)
        ]))
    elif len(params):
        return Universe(params)
    elif len(product):
        return Universe(tuple([
            frozendict({
                k: v
                for k, v in zip(product.keys(), vs)
            })
            for vs in itertools.product(
                *product.values()
            )
        ]))
    else:
        raise ValueError((params, product))


@dataclass(frozen=True)
class Filter:
    func: Callable[[Object, frozendict], bool] | None = None

    def __call__(self, obj: Object, info: frozendict) -> bool:
        if self.func is None:
            raise ValueError("Either pass a func or implement __call__.")
        return self.func(obj, info)


@dataclass(frozen=True)
class Predicate:
    filters: tuple[Filter, ...]
    values: t.frozen.dict = frozendict()

    name: str = ""

    group_by: tuple[str, ...] | None = None

    def apply(self):
        # TODO: remove bind as a class?
        # apply turns this into select
        # zip i guess needs to acc peers?
        # select is therefore .product() with others to get select
        return
    
    def product(self):
        # returns a sleect, with this, and peers
        return

    def zip(self):
        return self

    def group(self, *args: str):
        return replace(self, group_by=args)

    # NOTE: alias here is the kwarg this predicate will be fit under
    def alias(self, alias: str):
        return replace(self, name=alias)

    def generate(self, collection: Collection):
        if not len(self.filters) and not len(self.values):
            raise ValueError(self)
        
        objs = collection.where(
            info=self.values if len(self.values) else None,
            filters=self.filters if len(self.filters) else None,
        )
        if self.group_by is None:
            return [
                (self.name, k, obj)
                for k, obj in objs.items()
            ]
        elif not len(self.group_by):
            return [
                (self.name, list(objs.keys()), list(objs.values()))
            ]
        
        res = defaultdict(list)
        for k, obj in objs.items():
            info = collection.info.get(k, frozendict())
            vs = tuple([
                info.get(g, None) for g in self.group_by
            ])
            res[vs].append((k, obj))

        return [
            (self.name,) + tuple(map(list, zip(*k_objs)))
            for k_objs in res.values()
        ]
        
def filter(func: Callable[[Object, frozendict], bool]):
    return Filter(func)

def where(
    *filters: Filter | Callable[[Object, frozendict], bool],
    **values: Any,
) -> Predicate:
    return Predicate(tuple([
        f if isinstance(f, Filter) else Filter(f)
        for f in filters
    ]), frozendict(values))

@dataclass(frozen=True)
class Select:
    universes: tuple[Universe, ...]
    predicates: t.frozen.dict[str, Predicate]

    def generate(self, collection: Collection):
        for args in itertools.product(*[
            univ.generate(collection)
            for univ in self.universes
        ] + [
            predicate.generate(collection)
            for predicate in self.predicates.values()
        ]):
            yield args

    def apply(
        self,
        function: Callable,
        **params,
    ):
        return Apply(self, function, frozendict(params))
    def bind(
        self,
        function: Callable,
        alias: Callable | None = None,
        **params,
    ):
        return Bind(self, function, frozendict(params), name=alias)
    
    def output(
        self,
        function: Callable,
        **params,
    ):
        return Outputs(self, (
            Output(function, frozendict(params)),)
        )

def select(*universes: Universe, **predicates: Predicate):
    return Select(universes, frozendict({
        k: predicate.alias(k) for k, predicate in predicates.items()
    }))


@dataclass(frozen=True)
class Bind:
    # TODO: remove
    select: Select
    func: Callable
    params: t.frozen.dict

    name: Callable | None = None

    info: t.frozen.dict | None = None

    # TODO: list of fields to bind on info

    def alias(self, name: Callable):
        return replace(self, name=name)

    def __call__(self, collection: Collection):
        for args in self.select.generate(collection):
            kw = combine_args(self.params, args)
            params = bind_params(**kw)
            if self.name is None:
                raise ValueError(self)
            try:
                name = self.name(kw)
                bound = Bound(
                    self.func, 
                    name=name,
                    params=params
                ).with_info(
                    FUNC=self.func.__name__,
                    # name?
                    **kw,
                    **(
                        {} if self.info is None else self.info
                    )
                )
                collection = bound.bind(collection)
            except Exception as e:
                raise ValueError(kw, params) from e
        return collection


@dataclass(frozen=True)
class Apply:
    select: Select
    func: Callable
    params: t.frozen.dict

    name: Callable | None = None

    # TODO: list of fields to bind on info

    # TODO: optionally take a string first arg (ro seconr, then first is alwaus fcallable)
    # then the func is applied to kw[arg0]
    # NOTE: this is what we alias the results into the colleciton under
    def alias(self, name: Callable):
        return replace(self, name=name)

    def __call__(self, collection: Collection):
        for args in self.select.generate(collection):
            kw = combine_args(self.params, args)

            # TODO: add config name for info, added on __call__ by run.py

            # TODO: borrow ref resolution from bound

            # if self.name is None:
            #     raise ValueError(self)
            # name = self.name(kw)
            try:
                res = self.func(**kw)
            except Exception as e:
                raise ValueError(kw) from e
            # TODO: nested info - via dot path?
            if isinstance(res, Bound):
                collection = res.with_info(
                    FUNC=self.func,
                    **kw,
                    # TODO: add config name to info for selection
                ).bind(collection)
            else:
                raise ValueError(self, res)
        return collection
            

def combine_args(
    kw: frozendict,
    args: tuple[
        frozendict
        | tuple[str, str, Object]
        | tuple[str, list[str], list[object]]
    ]
):
    for arg in args:
        if isinstance(arg, frozendict):
            kw = kw | arg
        elif isinstance(arg, tuple):
            key, name, obj = arg
            kw = kw.set(key, (name, obj))
        else:
            raise ValueError(obj)
    return kw


@dataclass(frozen=True)
class Output:
    func: Callable
    params: t.frozen.dict

    def __call__(
        self,
        collection: Collection,
        args: tuple[
            frozendict
            | tuple[str, str, Object]
            | tuple[str, list[str], list[object]]
        ],
        acc: frozendict,
    ):
        kw = combine_args(self.params, args)
        acc = self.func(collection, acc, **kw)
        # TODO: res eg. just writes to app
        return acc


@dataclass(frozen=True)
class Outputs:
    select: Select
    outputs: tuple[Output, ...] = ()

    def __call__(
        self,
        collection: Collection, 
        acc: frozendict,
    ):
        for args in self.select.generate(collection):
            for output in self.outputs:
                acc = output(collection, args, acc)
        return acc

    def output(self, output: str, **params):
        return replace(self, outputs = self.outputs + (Output(
            output,
            frozendict(params)
        ),))
    
__all__ = [
    "Collection",
    "Bound",
    "bound",
    "init_bind",
    "bind_params",
    "unpack_bound",
    "universe",
    "where",
    "select",
    "Filter",
    "Output",
    "Outputs",
    "Select",
    "Bind",
    "Apply",
]