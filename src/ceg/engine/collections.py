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
    Generator,
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
from .daily import *


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
    aliases: t.frozen.dict[str, str] = frozendict()
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
        aliases=self.aliases
        objects = objects.set(key, obj)
        if isinstance(obj, Object):
            # TODO: composites like Strategy can also be bound
            aliases = aliases.set(obj.id, key)
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
            aliases=aliases,
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
    if isinstance(v, Aliased):
        return v.obj
    
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

# init_bind removed — was a factory for Bound-returning binders used only by
# now-deleted bind_sum / bind_prod / etc. helpers. Bound itself stays for
# legacy callers in engine/collections.py.

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
class Select:

    def generate(self, collection: Collection) -> Generator[dict]: ...
    
    def bind(
        self,
        func: Callable,
        **kwargs
    ):
        return Bind(self, func, frozendict(kwargs))
    
    def apply(
        self,
        func: Callable,
        **kwargs
    ):
        return Apply(self, (Output(func, frozendict(kwargs)),))
    
    def product(
        self,
        *others: Select,
        **named_others: Select,
    ):
        others = others + tuple([
            replace(other, name=name) for name, other in named_others.items()
        ])
        return SelectProduct.new(self, *others)

    def zip(
        self,
        *others: Select,
        **named_others: Select,
    ):
        others = others + tuple([
            replace(other, name=name) for name, other in named_others.items()
        ])
        return SelectZip.new(self, *others)

    def group(self, *args: str):
        return replace(self, group_by=args)

    # NOTE: alias here is the kwarg this predicate will be fit under
    def alias(self, alias: str):
        return replace(self, name=alias)

@dataclass(frozen=True)
class SelectLiteral(Select):
    params: tuple[t.frozen.dict, ...] = ()

    def add(self, **params):
        return replace(self, params = self.params + (frozendict(params),))
    
    def generate(self, collection: Collection):
        # TODO: apply group by
        return self.params

def zip_every_kwargs(d: dict[str, list], z: dict[str, list]):
    z_it = zip(*z.values())
    return tuple([
        frozendict({
            **{
                k: v for k, v in zip(z.keys(), vs[0])
            },
            **{
                k: v for k, v in zip(d.keys(), vs[1:])
            }
        })
        for vs in itertools.product(z_it, *d.values())
    ])

def merge_kwargs(*ds: dict):
    acc = {}
    for d in ds:
        assert not any([
            k in acc for k in d.keys()
        ]), (ds)
        acc = acc | d
    return acc
def every_kwargs(d: dict[str, list]):
    return tuple([
        frozendict({
            k: v for k, v in zip(d.keys(), vs)
        })
        for vs in itertools.product(*d.values())
    ])

def zip_kwargs(d: dict[str, list]):
    return tuple([
        frozendict({
            k: v for k, v in zip(d.keys(), vs)
        })
        for vs in zip(*d.values())
    ])


def select(
    *s: Select,
    every: dict[str, list] | bool | None = None,
    zip: dict[str, list] | bool | None = None,
    **where: Filter | dict | frozendict | list | Any | Select,
) -> SelectLiteral:
    if len(s):
        assert len(s) == 1, s
        # TODO: zip / every
        return s[0]

    b_every = isinstance(every, bool) and every
    b_zip = isinstance(zip, bool) and zip

    if b_every and b_zip:
        raise ValueError(every, zip)

    filters = {}

    for k, w in where.items():
        if isinstance(w, list):
            if zip is None:
                zip = {}
            zip[k] = w
        elif isinstance(w, Select):
            if not b_every and not b_zip:
                assert len(where) == 1, where
                # TODO: zip / ever
                return w.alias(k)
            else:
                filters[k] = w.alias(k)
        else:
            filters[k] = w

    if len(filters) and b_every:
        select = SelectProduct(tuple([
            f if isinstance(f, Select) else
            SelectWhere(
                () if not isinstance(f, Filter)
                else (f,),
                frozendict() if isinstance(f, Filter)
                else frozendict(f),
            ).alias(k)
            for k, f in filters.items()
        ]))
        if zip is not None:
            raise ValueError(filters, zip)
        return select
    elif len(filters) and b_zip:
        select = SelectZip(tuple([
            f if isinstance(f, Select) else
            SelectWhere(
                () if not isinstance(f, Filter)
                else (f,),
                frozendict() if isinstance(f, Filter)
                else frozendict(f),
            ).alias(k)
            for k, f in filters.items()
        ]))
        if every is not None:
            raise ValueError(filters, every)
        return select
    elif len(filters):
        select = SelectWhere(
            tuple([f for f in filters.values() if isinstance(f, Filter)]),
            values=frozendict({
                k: v for k, v in filters.items()
                if not isinstance(v, Filter)
                and not isinstance(v, dict)
            })
        )
        if any([isinstance(v, dict) for v in filters.values()]):
            raise ValueError(filters)
        if zip is not None or every is not None:
            raise ValueError(every, zip, where)
        return select

    if isinstance(every, bool) and every:
        if zip is None:
            raise ValueError(zip, every)
        every = zip
        zip = None

    if zip is not None and every is not None:
        return SelectLiteral(zip_every_kwargs(every, zip))
    elif zip is not None:
        return SelectLiteral(zip_kwargs(zip))
    elif every is not None:
        return SelectLiteral(every_kwargs(every))
    else:
        pass
    
    raise ValueError(every, zip, where)

@dataclass(frozen=True)
class Filter:
    func: Callable[[Object, frozendict], bool] | None = None

    def __call__(self, obj: Object, info: frozendict) -> bool:
        if self.func is None:
            raise ValueError("Either pass a func or implement __call__.")
        try:
            return self.func(obj, info)
        except Exception as e:
            raise ValueError(obj, info)


@dataclass(frozen=True)
class Aliased:
    name: str # in universe
    obj: Object | Any


@dataclass(frozen=True)
class SelectWhere(Select):
    filters: tuple[Filter, ...]
    values: t.frozen.dict = frozendict()

    name: str = ""

    group_by: tuple[str, ...] | None = None

    def generate(self, collection: Collection):
        if self.name == "":
            raise ValueError(self)
        
        if not len(self.filters) and not len(self.values):
            raise ValueError(self)
        
        objs = collection.where(
            info=self.values if len(self.values) else None,
            filters=self.filters if len(self.filters) else None,
        )
        if self.group_by is None:
            return [
                {self.name: Aliased(k, obj)}
                for k, obj in objs.items()
            ]
        elif not len(self.group_by):
            return [
                {self.name: [
                    Aliased(k, obj)
                    for k, obj in objs.items()
                ]}
            ]
        
        res = defaultdict(list)
        for k, obj in objs.items():
            info = collection.info.get(k, frozendict())
            vs = tuple([
                info.get(g, None) for g in self.group_by
            ])
            res[vs].append((k, obj))

        return [
            {self.name: [
                Aliased(k, obj) for k, obj in k_objs
            ]}
            for _, k_objs in res.items()
        ]
        


@dataclass(frozen=True)
class SelectProduct(Select):
    members: tuple[SelectLiteral | SelectWhere | SelectZip, ...]

    @classmethod
    def new(cls, *others: Select):
        acc = []
        for s in others:
            if isinstance(s, SelectProduct):
                acc.extend(s.members)
            else:
                acc.append(s)
        return cls(tuple(acc))
    
    def generate(self, coll: Collection):
        for kws in itertools.product(*[
            m.generate(coll) for m in self.members
        ]):
            yield merge_kwargs(*kws)


@dataclass(frozen=True)
class SelectZip(Select):
    members: tuple[SelectLiteral | SelectWhere | SelectZip, ...]

    def generate(self, coll: Collection):
        for kws in zip(*[
            m.generate(coll) for m in self.members
        ]):
            yield merge_kwargs(*kws)
        

    @classmethod
    def new(cls, *others: Select):
        acc = []
        for s in others:
            if isinstance(s, SelectZip):
                acc.extend(s.members)
            elif isinstance(s, SelectProduct):
                # TODO for now, not allowed?
                raise ValueError(others)
            else:
                acc.append(s)
        return cls(tuple(acc))

def filter(func: Callable[[Object, frozendict], bool]):
    return Filter(func)

def where(
    *filters: Filter | Callable[[Object, frozendict], bool],
    **values: Any,
) -> SelectWhere:
    return SelectWhere(tuple([
        f if isinstance(f, Filter) else Filter(f)
        for f in filters
    ]), frozendict(values))



@dataclass(frozen=True)
class Bind:
    select: Select
    func: Callable
    params: t.frozen.dict

    # TODO: config / apply name?

    name: Callable | str | None = None

    unpack_fields: tuple[str, ...] = ()
    arg_order: tuple[str, ...] = ()

    mutates_fields: tuple[str, ...] = ()
    bind_fields: tuple[str, ...] = ()

    into_fields: tuple[str, ...] = ()

    # NOTE: this is what we alias the results into the colleciton under
    def under(self, name: Callable | str):
        return replace(self, name=name)
    
    def unpack(self, *fields: str):
        return replace(self, unpack_fields=fields)
    
    def args(self, *args: str):
        return replace(self, arg_order=args)
    
    def mutates(self, *mutates: str):
        return replace(self, mutates_fields=mutates)
    
    def info(self, *bind: str):
        return replace(self, bind_fields=bind)
    
    def into(self, *bind: str):
        assert len(self.mutates_fields) == 0, self
        return replace(self, into_fields=bind).mutates(bind[0])

    def unpack_aliased(self, v):
        if isinstance(v, Aliased):
            return v.obj
        elif isinstance(v, list):
            return [self.unpack_aliased(vv) for vv in v]
        elif isinstance(v, tuple):
            return tuple([self.unpack_aliased(vv) for vv in v])
        # TODO dicts?
        else:
            return v

    def unpack_bound_kw(
        self,
        kw: dict,
        collection: Collection,
    ):
        params = {
            k: self.unpack_aliased(
                unpack_bound(v, k, self, collection, force=True)
                if k in self.unpack_fields
                else v if not isinstance(v, str)
                else unpack_bound(v, k, self, collection)
            )
            for k, v in kw.items()
        }

        args = self.arg_order
        if "parent" in params:
            args = ("parent,") + args

        return tuple([
            params[a] for a in args
        ]), {
            k: v for k, v in params.items() if k not in args
        }
    
    # TODO: list of fields to bind on info

    
    def alias_object(self, kw: dict):
        if isinstance(self.name, str):
            return self.name.format(**kw)
        return self.name(**kw)

    def __call__(self, collection: Collection, config: str):
        n = 0
        for args in self.select.generate(collection):
            kw = merge_kwargs(args, self.params)

            args, kw = self.unpack_bound_kw(kw, collection)

            if len(self.into_fields):
                into = tuple([
                    kw.pop(k) for k in self.into_fields
                ])
                into, *into_kw = into
                into_kw = dict(zip(
                    self.into_fields[1:], into_kw
                ))
            else:
                into = None

            if self.name is None:
                raise ValueError(self)
            
            try:
                res = self.func(**kw)
            except Exception as e:
                raise ValueError(kw) from e
            
            if into is not None:
                try:
                    res = res.bind(into, t=type(into), **into_kw)
                except Exception as e:
                    raise ValueError(into, res) from e
            
            if len(self.mutates_fields):
                *mutates, res = res
                # eg. if mutate container on object create, etc.
                for mut in mutates:
                    collection = replace(
                        collection,
                        objects=collection.objects.set(
                            collection.aliases[mut.id], mut
                        )
                    )

            # TODO: nested info - via dot path?
            # if isinstance(res, Object):

            alias = self.alias_object(kw)

            collection = collection.with_obj(
                res,
                alias,
                info = frozendict({
                    "FUNC": self.func,
                    "CONFIG": config,
                    **(
                        # kw
                        # if not len(self.bind_fields)
                        {
                            k: v for k, v in kw.items()
                            if k in self.bind_fields
                        }
                    )
                })
            )

            # elif isinstance(res, Bound):
            #     collection = res.with_info(
            #         # TODO: add config name to info for selection
            #     ).bind(collection)
            # else:
            #     raise ValueError(self, res, type(res))
            
            n += 1
        
        if n == 0:
            raise ValueError(self)
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

    then: tuple[Output, ...] = ()

    def with_then(self, output: Output):
        return replace(self, then=self.then + (output,))

    def __call__(
        self,
        collection: Collection,
        args: dict,
        #   tuple[
        #     frozendict
        #     | tuple[str, str, Object]
        #     | tuple[str, list[str], list[object]]
        # ],
        acc: frozendict,
        config: str,
    ):
        kw = merge_kwargs(args, self.params)
        pipe = (self,) + self.then

        for i, o in enumerate(pipe):
            if i == 0 and i == len(pipe) - 1:
                acc = o.func(collection, acc, **kw)
            elif i == 0:
                res = o.func(collection, **kw)
            elif i == len(pipe) - 1:
                if isinstance(res, dict):
                    acc = o.func(acc, **res, **o.params)
                else:
                    acc = o.func(res, acc, **o.params)
            else:
                if isinstance(res, dict):
                    res = o.func(**res, **o.params)
                else:
                    res = o.func(res, **o.params)
                
        return acc


@dataclass(frozen=True)
class Apply:
    select: Select
    outputs: tuple[Output, ...] = ()

    def __call__(
        self,
        collection: Collection, 
        acc: frozendict,
        config: str,
    ):
        n = 0
        for args in self.select.generate(collection):
            for output in self.outputs:
                acc = output(collection, args, acc, config=config)
            n += 1
        
        if n == 0:
            raise ValueError(self)
        return acc

    def apply(self, output: Callable, **params):
        return replace(self, outputs = self.outputs + (Output(
            output,
            frozendict(params)
        ),))
    
    def then(self, output: Callable, **params):
        return replace(self, outputs = (
            tuple(self.outputs[:-1])
            + (self.outputs[-1].with_then(Output(output, frozendict(params))),)
        ))
    
def unpack_daily_df(
    
    collection: Collection,

    objs: list[Aliased],

    unpack: str | None = None,

    # TODO: other_objs in kwargs, then expect keys  argument
):
    if unpack is None:
        df = daily_pl_df([
            al.obj for al in objs
        ]).rename({
            al.obj.id: al.name for al in objs
        })
    else:
        objs_unpacked = [
            collection.get(f"{al.name}.{unpack}()")
            for al in objs
        ]
        df = daily_pl_df(objs_unpacked).rename({
            obj.id: f"{al.name}.{unpack}" for al, obj in zip(
                objs,
                objs_unpacked
            )
        })
    return df
    
def unpack_daily_col_names(
    collection: Collection,
    objs: list[Aliased] | None = None,

    unpack: str | None = None,
    keys: list[str] | None = None,

    **named_objs: list[Aliased],
):
    if keys is not None:
        return {
            k: [
                (
                    al.name
                    if unpack is None 
                    else f"{al.name}.{unpack}"
                )
                for al in named_objs[k]
            ]
            for k in keys
        }
    
    return [
        (
            al.name
            if unpack is None 
            else f"{al.name}.{unpack}"
        )
        for al in objs
    ]
__all__ = [
    "Collection",
    "Bound",
    "bound",
    "bind_params",
    "unpack_bound",
    "where",
    "select",
    "Filter",
    "Apply",
    "Select",
    "Bind",
    "Aliased",
    "unpack_daily_df",
    "unpack_daily_col_names",
]