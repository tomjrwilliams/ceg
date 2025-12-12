
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

#  ------------------

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
class UniverseScale():
    index: tuple[int, ...]
    method: str | None = None
    scale: float = 1.
    activation: str | None = None

@dataclass(frozen=True)
class UniverseSignal():
    index: tuple[int, ...]
    method: str | None = None
    scale: float = 1.
    activation: str | None = None

@dataclass(frozen=True)
class UniversePosition(
):
    scale: int | UniverseScale
    signal: int | UniverseSignal | None=None
    rebalance: int | str | None=None
    calendar: int | None=None
    weight: int | None=None

    unit: float =100.

    index: int = -1

@dataclass(frozen=True)
class UniverseBt():
    pos: int
    rx: Object | str
    index: int = -1

@dataclass(frozen=True)
class UniverseAggregate():
    bt: tuple[int, ...]
    # name?
    index: int = -1

    # NOTE: if bt is empty, assume take all (for total portfolio, default)

@dataclass(frozen=True)
class BtParams:
    pass

@dataclass(frozen=True)
class BtValues:
    pass

@dataclass(frozen=True)
class BtResult:
    params: BtParams
    domain: Domain | None = None

    rx: str | None = None
    scale: str | tuple[str, ...] = ()
    signal: str | tuple[str, ...] | None = None
    rebalance: str | None = None
    # TODO: calendar
    # TODO: weight

    i_scale: int | UniverseScale = -1
    i_signal: int | UniverseSignal = -1
    i_rebalance: int = -1
    i_calendar: int = -1
    i_weight: int = -1

    i_pos: int = -1
    i_bt: int = -1

    res: BtValues | None = None
    # subtype with rx and pos

    def with_obj(
        self, obj: str | Object | tuple[str | Object, ...]
    ) -> BtResult:
        if self.domain is None:
            raise ValueError((obj, self))
        if isinstance(obj, str):
            return self
        elif isinstance(obj, Object):
            return replace(
                self, domain = self.domain.with_obj(obj)
            )
        for o in obj:
            self = self.with_obj(o)
        return self

    def done(self) -> tuple[BtResult, Domain]:
        domain = self.domain
        assert domain is not None, self
        return replace(self, domain=None), domain

TBtParams = TypeVar(
    "TBtParams", bound=BtParams
)
TBtResult = TypeVar(
    "TBtResult", bound=BtResult
)

@dataclass(frozen=True)
class Universe(
    UseRootConfigs,
):
    parent: EngineRoot | str

    name: str

    # TODO: daily chunked not just object?
    scale: tuple[Object | str, ...]
    signal: tuple[Object | str, ...] = ()
    rebalance: tuple[Object | str, ...] = ()
    calendar: tuple[Object | str, ...] = ()
    weight: tuple[Object | str, ...] = ()
    
    pos: tuple[UniversePosition, ...] = ()
    bt: tuple[UniverseBt, ...] = ()
    agg: tuple[UniverseAggregate, ...] = ()
    
    version: int = 0
    type: Literal["Universe"] = "Universe"

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, t=EngineRoot)
    
    @classmethod
    def new(
        cls,
        parent: EngineRoot,
        name: str,
        scale: tuple[Object | str, ...] = (),
        signal: tuple[Object | str, ...] = (),
        rebalance: tuple[Object | str, ...] = (),
        calendar: tuple[Object | str, ...] = (),
        weight: tuple[Object | str, ...] = (),
        pos: tuple[UniversePosition, ...] = (),
        bt: tuple[UniverseBt, ...] = (),
        agg: tuple[UniverseAggregate, ...] = (),
    ):
        return cls(
            parent=parent,
            path=f"p-{name}",
            name=name,
            scale=scale,
            signal=signal,
            rebalance=rebalance,
            calendar=calendar,
            weight=weight,
            pos=pos,
            bt=bt,
            agg=agg,
        )
    
    def with_object(
        self, obj: Object | str, field: str,
    ) -> tuple[Universe, int]:
        objs = getattr(self, field)
        i = find_object(obj, objs)
        if i is not None:
            return self, i
        objs = objs + (obj,)
        return (
            replace(self, **{field: objs}),
            len(objs) - 1
        )

    @overload
    def with_scale(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: BtResult,
        agg: str | None = None,
        scale: float | None = None,
        activation: str | None = None,
    ) -> tuple[
        Universe, 
        BtResult
    ]: ...
    @overload
    def with_scale(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: None = None,
        agg: str | None = None,
        scale: float | None = None,
        activation: str | None = None,
    ) -> tuple[
        Universe, 
        int | UniverseScale
    ]: ...
    def with_scale(
        self, 
        obj: Object | str | tuple[Object | str, ...],
        into: BtResult | None = None,
        agg: str | None = None,
        scale: float | None = None,
        activation: str | None = None,
    ) -> tuple[
        Universe, 
        int | UniverseScale | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, "scale")
        else:
            assert agg is not None, (self, agg)
            assert scale is not None, (self, scale)
            indices = []
            for sc in obj:
                self, i = self.with_scale(sc)
                indices.append(i)
            self, i = self, UniverseScale(
                tuple(indices),
                method=agg,
                scale=scale,
                activation=activation,
            )
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            scale=obj_path(obj),
            i_scale=i,
        )
    
    @overload
    def with_signal(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: BtResult,
        agg: str | None = None,
        scale: float | None = None,
        activation: str | None = None,
    ) -> tuple[Universe, BtResult]: ...
    @overload
    def with_signal(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: BtResult | None = None,
        agg: str | None = None,
        scale: float | None = None,
        activation: str | None = None,
    ) -> tuple[Universe, int | UniverseSignal]: ...
    def with_signal(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: BtResult | None = None,
        agg: str | None = None,
        scale: float | None = None,
        activation: str | None = None,
    ) -> tuple[
        Universe, int | UniverseSignal | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, "signal")
        else:
            assert agg is not None, (self, agg)
            assert scale is not None, (self, scale)
            indices = []
            for sc in obj:
                self, i = self.with_signal(sc)
                indices.append(i)
            self, i = self, UniverseSignal(
                tuple(indices),
                method=agg,
                scale=scale,
                activation=activation,
            )
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            signal=obj_path(obj),
            i_signal=i,
        )
    
    def with_rebalance(self, rebalance: Object | str) -> tuple[Universe, int]:
        return self.with_object(rebalance, "rebalance")
    
    def with_calendar(self, calendar: Object | str) -> tuple[Universe, int]:
        return self.with_object(calendar, "calendar")
    
    def with_weight(self, weight: Object | str) -> tuple[Universe, int]:
        return self.with_object(weight, "weight")
    
    @overload
    def with_pos(
        self, 
        pos: BtResult,
        signal: int | UniverseSignal | None=None,
        rebalance: int | str | None=None,
        calendar: int | None=None,
        weight: int | None=None,
        unit: float =100.,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_pos(
        self, 
        pos: (
            int 
            | UniverseScale 
            | UniversePosition
        ),
        signal: int | UniverseSignal | None=None,
        rebalance: int | str | None=None,
        calendar: int | None=None,
        weight: int | None=None,
        unit: float =100.,
    ) -> tuple[Universe, int]: ...

    def with_pos(
        self, 
        pos: (
            BtResult 
            | int 
            | UniverseScale 
            | UniversePosition
        ),
        signal: int | UniverseSignal | None=None,
        rebalance: int | str | None=None,
        calendar: int | None=None,
        weight: int | None=None,
        unit: float =100.,
    ) -> tuple[Universe, BtResult | int]:
        i = len(self.pos)
        res: BtResult | int = i
        if isinstance(pos, BtResult):
            scale = pos.i_scale
            assert signal is None, signal
            signal = pos.i_signal
            rebalance = rebalance or pos.rebalance
            res = replace(pos, i_pos=i)
            pos = UniversePosition(
                scale=scale,
                signal=signal,
                rebalance=rebalance,
                calendar=calendar,
                weight=weight,
                unit=unit,
                index=i,
            )
        elif isinstance(pos, UniversePosition):
            # assert scale is None, scale
            pos = replace(pos, index=i)
        else:
            if not isinstance(pos, int):
                scale = UniverseScale(
                    pos.index, pos.method
                )
            pos = UniversePosition(
                scale=scale,
                signal=signal,
                rebalance=rebalance,
                calendar=calendar,
                weight=weight,
                unit=unit,
                index=i,
            )
        return replace(self, pos=self.pos + (pos,)), res
    
    @overload
    def with_bt(
        self, 
        pos: BtResult,
        rx: Object | str,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_bt(
        self, 
        pos: int | UniverseBt,
        rx: Object | str,
    ) -> tuple[Universe, int]: ...

    def with_bt(
        self, 
        pos: int | UniverseBt | BtResult,
        rx: Object | str,
    ) -> tuple[Universe, BtResult | int]:
        i = len(self.bt)
        res: BtResult | int = i
        if isinstance(pos, BtResult):
            assert pos.i_pos is not None, pos
            bt = UniverseBt(pos.i_pos, rx, i)
            pos = pos.with_obj(rx)
            res = self.finish_bt(replace(pos, i_bt = i))
        elif isinstance(pos, UniverseBt):
            bt = replace(pos, index=i)
        else:
            bt = UniverseBt(pos, rx, i)
        return replace(self, bt=self.bt + (bt,)), res
    
    def finish_bt(self, res: BtResult) -> BtResult:
        return res

    def with_agg(
        self, 
        agg: tuple[int, ...] | UniverseAggregate,
    ):
        i = len(self.agg)
        if isinstance(agg, UniverseAggregate):
            assert agg is None, agg
            agg = replace(agg, index=i)
        else:
            agg = UniverseAggregate(agg, i)
        return replace(self, agg=self.agg + (agg,)), i
    
    def get_scale(
        self, i: int, domain: Domain, t: Type[O]
    ) -> O:
        return domain.get(self.scale[i], t=t)

    def get_signal(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.signal is None:
            return None
        return domain.get(self.signal[i], t=t)
    
    def get_rebalance(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.rebalance is None:
            return None
        return domain.get(self.rebalance[i], t=t)
    
    def get_calendar(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.calendar is None:
            return None
        return domain.get(self.calendar[i], t=t)
    
    def get_weight(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.weight is None:
            return None
        return domain.get(self.weight[i], t=t)

#  ------------------

__all__ = [
    "Universe",
    "UniverseScale",
    "UniverseSignal",
    "UniversePosition",
    "UniverseBt",
    "UniverseAggregate",
    "BtValues",
    "TBtParams",
    "BtParams",
    "TBtResult",
    "BtResult",
]