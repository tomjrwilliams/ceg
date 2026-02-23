
from __future__ import annotations
import os
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
import datetime
import datetime as dt

import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import polars as pl

from .... import apis
from ....apis import t, kv, assert_is
from ....engine import *

@dataclass(frozen=True)
class Aggregate():    
    objs: tuple[int, ...]
    # name?
    index: int = -1

    # NOTE: if objs is empty, assume take all (for eg. total portfolio?)

    # TODO: with_obj (returns a new, with it added)

    def get_objects(
        self,
        collection: Collection,
    ):
        if len(self.objs) == 0:
            raise ValueError(self)
            # objs = [
            #     collection.get_object(
            #         i, Daily
            #     )
            #     for i in range(
            #         len(collection.objects)
            #     )           
            # ]
        else:
            objs = [
                collection.get_object(
                    i, Daily
                )
                for i in self.objs
            ]
        return objs

@dataclass(frozen=True)
class Ref():
    type: Literal["obj"] | Literal["agg"]
    i: int

    def encode(self):
        return f"{self.type}_{self.i}"
    
    @classmethod
    def decode(cls, s: str):
        t, i = s.split("_", maxsplit=1)
        assert t == "obj" or t == "agg", s
        return Ref(t, int(i))

TParamValue = bool | int | float | str | None
TParamDict = t.frozen.dict[str, TParamValue]

@dataclass(frozen=True)
class Collection(
    TracksLatestUpstream,
    UseRootStorage,
):
    parent: Project | str 

    name: str

    objects: tuple[Object | str, ...]
    agg: tuple[Aggregate, ...] = ()
    comb: tuple[Combination, ...] = ()

    params: tuple[TParamDict, ...] = ()

    type: Literal["Collection"] = "Collection"

    @dataclass(frozen=True)
    class Snap(TracksLatestUpstream.Snap):
        type: Literal["CollectionSnap"] = "CollectionSnap"

    @classmethod
    def new(
        cls,
        parent: Project,
        name: str,
        objects: tuple[Object | str, ...] = (),
        agg: tuple[Aggregate, ...] = (),
        comb: tuple[Combination, ...] = (),
        params: tuple[TParamDict, ...] = (),
    ):
        return cls(
            parent=parent,
            path=parent.extend_path(name),
            name=name,
            objects=objects,
            agg=agg,
            comb=comb,
            params=params,
        )
    
    def assume_object(self, i: int, t: Type[O]) -> O:
        o = self.objects[i]
        assert isinstance(o, t), (o, t)
        return o
    
    def upstream_paths(self):
        upstream = set()

        for obj in self.objects:
            if obj_path(obj) == self.path:
                raise ValueError((obj, self))
            upstream.add(obj_path(obj))

        return list(upstream)

    def depends(self) -> list[Dependency]:
        return [
            Dependency(
                self.path,
                depend,
                events=("ValueUpdate",)
            )
            for depend in self.upstream_paths()
            if depend != self.path
        ]
    
    def handler(
        self, event: Event, snap: Snap
    ) -> Handler:
        if isinstance(event, ValueUpdate):
            # do the check if ready to tick here?
            return HandleUpdateLatestUpstream()
        else:
            raise ValueError((self, event))
        
    @overload
    def with_params(
        self, 
        t: Type[TCollection],
        obj: TParamDict | None=None,
        **kwargs: TParamValue
    ) -> tuple[TCollection, int]: ...

    @overload
    def with_params(
        self,
        t: None = None,
        obj: TParamDict | None=None,
        **kwargs: TParamValue
    ) -> tuple[Collection, int]: ...

    def with_params(
        self,
        t: Type[TCollection] | None = None,
        obj: TParamDict | None=None,
        **kwargs: TParamValue
    ) -> tuple[Collection | TCollection, int]:
        params = frozendict({
            **({} if obj is None else obj),
            **kwargs
        })
        try:
            i = self.params.index(params)
        except ValueError:
            i = len(self.params)
            self = replace(
                self, params= self.params + (params,)
            )
        if t is not None:
            return assert_is(self, t), i
        return self, i

    @overload
    def with_object(
        self, obj: Object | str, t: Type[TCollection],
    ) -> tuple[TCollection, int]: ...

    @overload
    def with_object(
        self, obj: Object | str, t: None = None,
    ) -> tuple[Collection, int]: ...

    def with_object(
        self, obj: Object | str, t: Type[TCollection] |None = None,
    ) -> tuple[Collection | TCollection, int]:
        assert obj_path(obj) != self.path, dict(
            obj=(
                f"{type(obj).__name__}"
                f"({obj_path(obj)})"
            ),
            self=(
                f"{type(self).__name__}"
                f"({self.path})"
            ),
        )
        i = find_object(obj, self.objects)
        if i is not None:
            if t is not None:
                assert isinstance(self, t), (self, t)
            return self, i
        objs = self.objects + (obj,)
        self = replace(self, objects=objs)
        if t is not None:
            assert isinstance(self, t), (self, t)
        return (
            self,    
            len(objs) - 1
        )

    @overload
    def with_agg(
        self, 
        agg: tuple[Object | str | int, ...] | Aggregate,
        t: Type[TCollection] ,
    ) -> tuple[TCollection, Aggregate]: ...
        
    @overload
    def with_agg(
        self, 
        agg: tuple[Object | str | int, ...] | Aggregate,
        t: None = None,
    ) -> tuple[Collection, Aggregate]: ...

    def with_agg(
        self, 
        agg: tuple[Object | str | int, ...] | Aggregate,
        t: Type[TCollection] | None = None,
    ) -> tuple[Collection | TCollection, Aggregate]:
        i_agg = len(self.agg)
        # TODO: if agg in aggs
        if isinstance(agg, Aggregate):
            res = replace(agg, index=i_agg)
        else:
            args = []
            for o in agg:
                if isinstance(o, Ref):
                    assert o.type == "obj", o
                    args.append(o.i)
                elif isinstance(o, int):
                    args.append(o)
                else:
                    if isinstance(o, Object):
                        context.get().with_obj(o)
                    self, i = (
                        self.with_object(o)
                    )
                    args.append(i)
            res = Aggregate(tuple(args), i_agg)
        self = replace(self, agg=self.agg + (res,))
        if t is not None:
            return assert_is(self, t), res
        return self, res
    
    @overload
    def with_comb(
        self, 
        comb: dict[
            str,
            int
            | Object
            | str
            | Aggregate
            | tuple[int | Object | str, ...]
            | Ref
        ] | Combination,
        t: Type[TCollection] ,
    ) -> tuple[TCollection, Combination]: ...
    
    @overload
    def with_comb(
        self, 
        comb: dict[
            str,
            int
            | Object
            | str
            | Aggregate
            | tuple[int | Object | str, ...]
            | Ref
        ] | Combination,
        t: None = None,
    ) -> tuple[Collection, Combination]: ...
    
    def with_comb(
        self, 
        comb: dict[
            str,
            int
            | Object
            | str
            | Aggregate
            | tuple[int | Object | str, ...]
            | Ref
        ] | Combination,
        t: Type[TCollection] | None = None,
    ):
        i_comb = len(self.comb)
        if isinstance(comb, Combination):
            res = replace(comb, index=i_comb)
        else:
            res = Combination().with_objects(
                self,
                t=Collection,
                index=i_comb,
                **comb,
            )
        self = replace(self, comb=self.comb + (res,))
        if t is not None:
            return assert_is(self, t), res
        return self, res
    
    @overload
    def get_object(
        self, i: int, t: Type[O]
    ) -> O: ...
    @overload
    def get_object(
        self, i: None, t: Type[O]
    ) -> None: ...
    def get_object(
        self, i: int | None, t: Type[O]
    ) -> O | None:
        if i is None:
            return None
        return context.get().get(self.objects[i], t=t)

TCollection = TypeVar("TCollection", bound=Collection)

@dataclass(frozen=True)
class Combination():
    refs: t.frozen.dict[str, str] = frozendict()

    key: str | None = None

    params: int | None = None

    index: int = -1 # of self

    def get_obj(
        self,#
        key: str,
        collection: Collection,
        t: Type[O]
    ) -> O:
        ref = Ref.decode(
            assert_is(self.refs.get(key, None))
        )
        assert ref.type == "obj", ref
        return context.get().get(
            collection.objects[ref.i], t
        )
    
    def with_params(
        self,
        collection: Collection,
        t: Type[TCollection] = Collection,
        **kwargs: TParamValue,
    ):
        coll, i = collection.with_params(
            t=t, obj=None, **kwargs
        )
        self = replace(self, params=i)
        return coll, self

    def with_objects(
        self,
        collection: Collection,
        t: Type[TCollection] = Collection,
        **objs: (
            int
            | str
            | Object
            | Aggregate
            | tuple[int | str | Object, ...]
            | Ref
        ),
    ):
        refs = self.refs
        for key, obj in objs.items():
            if isinstance(obj, Ref):
                # assert in collection?
                refs = refs.set(key, obj.encode())
            elif isinstance(obj, int):
                # assert in collection?
                refs = refs.set(key, Ref("obj", i).encode())
            elif isinstance(obj, Aggregate):
                assert obj.index > -1, agg
                refs = refs.set(key, Ref("agg", obj.index).encode())
            elif isinstance(obj, tuple):
                collection, agg = collection.with_agg(obj)
                refs = refs.set(key, Ref("agg", agg.index).encode())
            else:
                if isinstance(obj, Object):
                    context.get().with_obj(obj)
                collection, i = collection.with_object(obj)
                refs = refs.set(key, Ref("obj", i).encode())
        return (
            replace(self, refs=refs),
            assert_is(collection, t)
        )
    
__all__ = [
    "Ref",
    "Aggregate",
    "Collection",
    "Combination",
    "TCollection",
    "TParamValue",
    "TParamDict",
]