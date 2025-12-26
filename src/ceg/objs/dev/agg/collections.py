
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
from ....apis import t
from ....apis import kv
from ....engine import *

@dataclass(frozen=True)
class Aggregate():
    objs: tuple[int, ...]
    # name?
    index: int = -1

    # NOTE: if bt is empty, assume take all (for total portfolio, default)

@dataclass(frozen=True)
class Collection(
    TracksLatestUpstream,
    UseRootConfigs,
):
    parent: EngineRoot | str

    name: str

    objects: tuple[Object | str, ...]

    agg: tuple[Aggregate, ...] = ()

    type: Literal["Collection"] = "Collection"

    @dataclass(frozen=True)
    class Snap(TracksLatestUpstream.Snap):
        type: Literal["CollectionSnap"] = "CollectionSnap"

    @classmethod
    def new(
        cls,
        parent: EngineRoot,
        name: str,
        objects: tuple[Object | str, ...] = (),
        agg: tuple[Aggregate, ...] = (),
    ):
        return cls(
            parent=parent,
            path=name,
            name=name,
            objects=objects,
            agg=agg,
        )
    
    def assume_object(self, i: int, t: Type[O]) -> O:
        o = self.objects[i]
        assert isinstance(o, t), (o, t)
        return o
    
    def upstream_paths(self):
        upstream = set()

        for obj in self.objects:
            upstream.add(obj_path(obj))

        return list(upstream)

    def depends(self, domain: Domain) -> list[Dependency]:
        return [
            Dependency(
                self.path,
                depend,
                events=("ValueUpdate",)
            )
            for depend in self.upstream_paths()
        ]
    
    def handlers(
        self, event: Event, domain: Domain
    ) -> list[Handler]:
        if isinstance(event, ValueUpdate):
            return [
                HandleUpdateLatestUpstream()
            ]
        else:
            raise ValueError((self, event))
        
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

    def with_agg(
        self, 
        agg: tuple[int, ...] | Aggregate,
    ):
        i = len(self.agg)
        if isinstance(agg, Aggregate):
            assert agg is None, agg
            agg = replace(agg, index=i)
        else:
            agg = Aggregate(agg, i)
        return replace(self, agg=self.agg + (agg,)), i
    
    @overload
    def get_object(
        self, i: int, domain: Domain, t: Type[O]
    ) -> O: ...
    @overload
    def get_object(
        self, i: None, domain: Domain, t: Type[O]
    ) -> None: ...
    def get_object(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None:
            return None
        return domain.get(self.objects[i], t=t)

TCollection = TypeVar("TCollection", bound=Collection)

__all__ =[
    "Aggregate",
    "Collection",
    "TCollection",
]