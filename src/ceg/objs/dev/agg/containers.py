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
class Aggregate:
    objs: tuple[int, ...]
    # name?
    index: int = -1

    
    @classmethod
    def obj(cls, i: int):
        return cls("obj", i)
    @classmethod
    def agg(cls, i: int):
        return cls("agg", i)

    # NOTE: if objs is empty, assume take all (for eg. total portfolio?)

    # TODO: with_obj (returns a new, with it added)

    def get_objects(
        self,
        container: Container,
    ):
        if len(self.objs) == 0:
            raise ValueError(self)
            # objs = [
            #     container.get_object(
            #         i, Daily
            #     )
            #     for i in range(
            #         len(container.objects)
            #     )
            # ]
        else:
            objs = [
                container.get_object(i, Daily)
                for i in self.objs
            ]
        return objs


@dataclass(frozen=True)
class Ref:
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
class Container(
    TriageUpstream,
    UseRootStorage,
):
    parent: Project | str

    name: str

    objects: tuple[Object | str, ...]
    agg: tuple[Aggregate, ...] = ()
    comb: tuple[Combination, ...] = ()

    params: tuple[TParamDict, ...] = ()

    type: Literal["Container"] = "Container"

    @dataclass(frozen=True)
    class Snap(TriageUpstream.Snap):
        type: Literal["ContainerSnap"] = "ContainerSnap"

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
        if isinstance(parent, str):
            parent = context.get().get(parent, Project)
        return cls(
            parent=parent,
            id=parent.extend_id(name),
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
            if obj_path(obj) == self.id:
                raise ValueError((obj, self))
            upstream.add(obj_path(obj))

        return list(upstream)

    def depends(self) -> list[Dependency]:
        return [
            Dependency(
                self.id, depend, events=("ValueUpdate",)
            )
            for depend in self.upstream_paths()
            if depend != self.id
        ]

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            # do the check if ready to tick here?
            return HandleUpdateLatestUpstream()
        else:
            raise ValueError((self, event))

    @overload
    def with_params(
        self,
        t: Type[TContainer],
        obj: TParamDict | None = None,
        **kwargs: TParamValue,
    ) -> tuple[TContainer, int]: ...

    @overload
    def with_params(
        self,
        t: None = None,
        obj: TParamDict | None = None,
        **kwargs: TParamValue,
    ) -> tuple[Container, int]: ...

    def with_params(
        self,
        t: Type[TContainer] | None = None,
        obj: TParamDict | None = None,
        **kwargs: TParamValue,
    ) -> tuple[Container | TContainer, int]:
        params = frozendict(
            {**({} if obj is None else obj), **kwargs}
        )
        try:
            i = self.params.index(params)
        except ValueError:
            i = len(self.params)
            self = replace(
                self, params=self.params + (params,)
            )
        if t is not None:
            return assert_is(self, t), i
        return self, i

    @overload
    def with_object(
        self,
        obj: Object | str,
        t: Type[TContainer],
    ) -> tuple[TContainer, int]: ...

    @overload
    def with_object(
        self,
        obj: Object | str,
        t: None = None,
    ) -> tuple[Container, int]: ...

    def with_object(
        self,
        obj: Object | str,
        t: Type[TContainer] | None = None,
    ) -> tuple[Container | TContainer, int]:
        assert obj_path(obj) != self.id, dict(
            obj=(f"{type(obj).__name__}({obj_path(obj)})"),
            self=(f"{type(self).__name__}({self.id})"),
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
        return (self, len(objs) - 1)

    @overload
    def with_agg(
        self,
        agg: tuple[Object | str | int, ...] | Aggregate,
        t: Type[TContainer],
    ) -> tuple[TContainer, Aggregate]: ...

    @overload
    def with_agg(
        self,
        agg: tuple[Object | str | int, ...] | Aggregate,
        t: None = None,
    ) -> tuple[Container, Aggregate]: ...

    def with_agg(
        self,
        agg: tuple[Object | str | int, ...] | Aggregate,
        t: Type[TContainer] | None = None,
    ) -> tuple[Container | TContainer, Aggregate]:
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
                    self, i = self.with_object(o)
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
            | Ref,
        ]
        | Combination,
        t: Type[TContainer],
    ) -> tuple[TContainer, Combination]: ...

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
            | Ref,
        ]
        | Combination,
        t: None = None,
    ) -> tuple[Container, Combination]: ...

    def with_comb(
        self,
        comb: dict[
            str,
            int
            | Object
            | str
            | Aggregate
            | tuple[int | Object | str, ...]
            | Ref,
        ]
        | Combination,
        t: Type[TContainer] | None = None,
    ):
        i_comb = len(self.comb)
        if isinstance(comb, Combination):
            res = replace(comb, index=i_comb)
        else:
            res = Combination().with_objects(
                self,
                t=Container,
                index=i_comb,
                **comb,
            )
        self = replace(self, comb=self.comb + (res,))
        if t is not None:
            return assert_is(self, t), res
        return self, res

    @overload
    def get_object(self, i: int, t: Type[O]) -> O: ...
    @overload
    def get_object(self, i: None, t: Type[O]) -> None: ...
    def get_object(
        self, i: int | None, t: Type[O]
    ) -> O | None:
        if i is None:
            return None
        return context.get().get(self.objects[i], t=t)


TContainer = TypeVar("TContainer", bound=Container)


@dataclass(frozen=True)
class Combination:
    refs: t.frozen.dict[str, str] = frozendict()

    key: str | None = None

    params: int | None = None

    index: int = -1  # of self

    def get_obj(
        self,  #
        key: str,
        container: Container,
        t: Type[O],
    ) -> O:
        ref = Ref.decode(
            assert_is(self.refs.get(key, None))
        )
        assert ref.type == "obj", ref
        return context.get().get(
            container.objects[ref.i], t
        )

    def with_params(
        self,
        container: Container,
        t: Type[TContainer] = Container,
        **kwargs: TParamValue,
    ):
        coll, i = container.with_params(
            t=t, obj=None, **kwargs
        )
        self = replace(self, params=i)
        return coll, self

    def with_objects(
        self,
        container: Container,
        t: Type[TContainer] = Container,
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
                # assert in container?
                refs = refs.set(key, obj.encode())
            elif isinstance(obj, int):
                # assert in container?
                refs = refs.set(key, Ref("obj", i).encode())
            elif isinstance(obj, Aggregate):
                assert obj.index > -1, agg
                refs = refs.set(
                    key, Ref("agg", obj.index).encode()
                )
            elif isinstance(obj, tuple):
                container, agg = container.with_agg(obj)
                refs = refs.set(
                    key, Ref("agg", agg.index).encode()
                )
            else:
                if isinstance(obj, Object):
                    context.get().with_obj(obj)
                container, i = container.with_object(obj)
                refs = refs.set(key, Ref("obj", i).encode())
        return (
            replace(self, refs=refs),
            assert_is(container, t),
        )


__all__ = [
    "Ref",
    "Aggregate",
    "Container",
    "Combination",
    "TContainer",
    "TParamValue",
    "TParamDict",
]
