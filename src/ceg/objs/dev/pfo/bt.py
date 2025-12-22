
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

from ..agg import *

#  ------------------


@dataclass(frozen=True)
class BtScale():
    index: tuple[int, ...]
    method: str | None = None
    scale: float = 1.

@dataclass(frozen=True)
class BtSignal():
    index: tuple[int, ...]
    method: str | None = None
    scale: float = 1.

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

    # TODO: concrete key
    key: str | None = None

    cap: str | None = None
    # TODO: calendar
    # TODO: weight

    i_scale: int | BtScale = -1
    i_rx: int = -1
    i_signal: int | BtSignal | None = None
    i_rebalance: int | None = None
    i_calendar: int | None = None
    i_weight: int | None = None
    i_cap: int | None = None
    i_fx: int | None = None

    i: int | None = None

    res: BtValues | None = None
    # subtype with rx and pos


    # TODO: chaining opperation, start new, acc ids
    # eg. .done_next()

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
class Backtest():
    scale: int | BtScale
    rx: int | None = None
    signal: int | BtSignal | None=None
    rebalance: int | str | None = None
    calendar: int | None=None
    weight: int | None=None
    cap: int | None = None

    fx: int | None = None

    # NOTE: concrete key for summation
    key: str | None = None

    unit: float =100.

    index: int = -1

    def validate(self):
        if isinstance(self.scale, int):
            assert self.scale >= 0, self
        if isinstance(self.signal, int):
            assert self.signal >= 0, self
        if isinstance(self.rebalance, int):
            assert self.rebalance >= 0, self
        if isinstance(self.calendar, int):
            assert self.calendar >= 0, self
        if isinstance(self.weight, int):
            assert self.weight >= 0, self
        if isinstance(self.cap, int):
            assert self.cap >= 0, self
        if isinstance(self.rx, int):
            assert self.rx >= 0, self
        if isinstance(self.fx, int):
            assert self.fx >= 0, self
        return self

    def get_scale(self) -> tuple[
        int | tuple[int, ...], float, str | None
    ]:
        if isinstance(self.scale, int):
            return self.scale, 1., None
        return (
            self.scale.index,
            self.scale.scale,
            self.scale.method or "prod"
        )
    
    def get_signal(self) -> tuple[
        int | tuple[int, ...], float, str | None
    ]:
        assert self.signal is not None, self
        if isinstance(self.signal, int):
            return self.signal, 1., None
        return (
            self.signal.index,
            self.signal.scale,
            self.signal.method or "prod"
        )

@dataclass(frozen=True)
class BtUvar():
    i: int
    
@dataclass(frozen=True)
class BtMvar():
    i: tuple[int, ...]
    

# TODO: rename BtUniverse?
@dataclass(frozen=True)
class Universe(Collection):
    
    bt: tuple[Backtest, ...] = ()
    uvar: tuple[BtUvar, ...] = ()
    mvar: tuple[BtMvar, ...] = ()
    
    version: int = 0
    type: Literal["Universe"] = "Universe"

    @dataclass(frozen=True)
    class Snap(TracksLatestUpstream.Snap):
        type: Literal["UniverseSnap"] = "UniverseSnap"

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, t=EngineRoot)
    
    @classmethod
    def new(
        cls,
        parent: EngineRoot,
        name: str,
        objects: tuple[Object | str, ...] = (),
        agg: tuple[Aggregate, ...] = (),
        bt: tuple[Backtest, ...] = (),
        uvar: tuple[BtUvar, ...] = (),
        mvar: tuple[BtMvar, ...] = (),
    ):
        return cls(
            parent=parent,
            path=name,
            name=name,
            objects=objects,
            agg=agg,
            bt=bt,
            uvar=uvar,
            mvar=mvar,
        )
    
    def with_uvar(self, bt: BtUvar):
        self = replace(self, uvar=self.uvar + (bt,))
        return self
    
    @overload
    def with_scale(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: BtResult,
        agg: str | None = None,
        scale: float | None = None,
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
    ) -> tuple[
        Universe, 
        int | BtScale
    ]: ...

    def with_scale(
        self, 
        obj: Object | str | tuple[Object | str, ...],
        into: BtResult | None = None,
        agg: str | None = None,
        scale: float | None = None,
    ) -> tuple[
        Universe, 
        int | BtScale | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, t = Universe)
            if scale is not None:
                assert scale is not None, (scale, obj)
                i = BtScale(
                    (i,),
                    method=agg,
                    scale=scale,
                )
        else:
            assert agg is not None, (self, agg)
            assert scale is not None, (self, scale)
            indices = []
            for sc in obj:
                self, i = self.with_scale(sc)
                indices.append(i)
            self, i = self, BtScale(
                tuple(indices),
                method=agg,
                scale=scale,
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
    ) -> tuple[Universe, BtResult]: ...
    @overload
    def with_signal(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: BtResult | None = None,
        agg: str | None = None,
        scale: float | None = None,
    ) -> tuple[Universe, int | BtSignal]: ...
    def with_signal(
        self, 
        obj: Object | str | tuple[Object, ...],
        into: BtResult | None = None,
        agg: str | None = None,
        scale: float | None = None,
    ) -> tuple[
        Universe, int | BtSignal | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, t= Universe)
            if scale is not None:
                i = BtSignal(
                    (i,),
                    method=agg,
                    scale=scale,
                )
        else:
            assert agg is not None, (self, agg)
            assert scale is not None, (self, scale)
            indices = []
            for sc in obj:
                self, i = self.with_signal(sc)
                indices.append(i)
            self, i = self, BtSignal(
                tuple(indices),
                method=agg,
                scale=scale,
            )
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            signal=obj_path(obj),
            i_signal=i,
        )
    
    @overload
    def with_rebalance(
        self, 
        obj: Object | str,
        into: BtResult ,
        literal: bool = False,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_rebalance(
        self, 
        obj: Object | str,
        into: None = None,
        literal: bool = False,
    ) -> tuple[Universe, int]: ...

    def with_rebalance(
        self, 
        obj: Object | str,
        into: BtResult | None = None,
        literal: bool = False,
    ) -> tuple[
        Universe, int | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, t= Universe)
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            rebalance=obj_path(obj),
            i_rebalance=i,
        )
    
    @overload
    def with_calendar(
        self, 
        obj: Object | str,
        into: BtResult,
        literal: bool = False,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_calendar(
        self, 
        obj: Object | str,
        into: None = None,
        literal: bool = False,
    ) -> tuple[Universe, int]: ...

    def with_calendar(
        self, 
        obj: Object | str,
        into: BtResult | None = None,
        literal: bool = False,
    ) -> tuple[
        Universe, int | BtResult
    ]:
        if isinstance(obj, (Object, str)) and not literal:
            self, i = self.with_object(obj, t= Universe)
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            calendar=obj_path(obj),
            i_calendar=i,
        )
    
    @overload
    def with_weight(
        self, 
        obj: Object | str,
        into: BtResult ,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_weight(
        self, 
        obj: Object | str,
        into: None = None,
    ) -> tuple[Universe, int]: ...

    def with_weight(
        self, 
        obj: Object | str,
        into: BtResult | None = None,
    ) -> tuple[
        Universe, int | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, t= Universe)
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            weight=obj_path(obj),
            i_weight=i,
        )
    
    @overload
    def with_rx(
        self, 
        obj: Object | str,
        into: BtResult ,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_rx(
        self, 
        obj: Object | str,
        into: None = None,
    ) -> tuple[Universe, int]: ...

    def with_rx(
        self, 
        obj: Object | str,
        into: BtResult | None = None,
    ) -> tuple[
        Universe, int | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, t= Universe)
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            rx=obj_path(obj),
            i_rx=i,
        )
    
    @overload
    def with_cap(
        self, 
        obj: Object | str,
        into: BtResult ,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_cap(
        self, 
        obj: Object | str,
        into: None = None,
    ) -> tuple[Universe, int]: ...

    def with_cap(
        self, 
        obj: Object | str,
        into: BtResult | None = None,
    ) -> tuple[
        Universe, int | BtResult
    ]:
        if isinstance(obj, (Object, str)):
            self, i = self.with_object(obj, t= Universe)
        if into is None:
            return self, i
        return self, replace(
            into.with_obj(obj),
            cap=obj_path(obj),
            i_cap=i,
        )
    
    @overload
    def with_bt(
        self, 
        bt: BtResult,
        rx: int | None = None,
        signal: int | BtSignal | None=None,
        rebalance: int | str | None=None,
        calendar: int | None=None,
        weight: int | None=None,
        cap: int | None=None,
        fx: int | None = None,
        unit: float =100.,
        key: str | None = None,
    ) -> tuple[Universe, BtResult]: ...

    @overload
    def with_bt(
        self, 
        bt: (
            int 
            | BtScale 
            | Backtest
        ),
        rx: int | None = None,
        signal: int | BtSignal | None=None,
        rebalance: int | str | None=None,
        calendar: int | None=None,
        weight: int | None=None,
        cap: int | None=None,
        fx: int | None = None,
        unit: float =100.,
        key: str | None = None,
    ) -> tuple[Universe, int]: ...

    def with_bt(
        self, 
        bt: (
            BtResult 
            | int 
            | BtScale 
            | Backtest
        ),
        rx: int | None = None,
        signal: int | BtSignal | None=None,
        rebalance: int | str | None=None,
        calendar: int | None=None,
        weight: int | None=None,
        cap: int | None=None,
        fx: int | None = None,
        unit: float =100.,
        key: str | None = None,
    ) -> tuple[Universe, BtResult | int]:
        i = len(self.bt)
        res: BtResult | int = i
        if isinstance(bt, BtResult):
            scale = bt.i_scale
            assert signal is None, signal
            signal = bt.i_signal
            rebalance = rebalance or bt.rebalance
            cap = cap or bt.i_cap
            # TODO: weight
            fx = fx or bt.i_cap
            rx = rx or bt.i_rx
            res = replace(bt, i=i)
            key = key or bt.key
            bt = Backtest(
                scale=scale,
                rx = rx,
                signal=signal,
                rebalance=rebalance,
                calendar=calendar,
                weight=weight,
                cap=cap,
                unit=unit,
                index=i,
                key=key,
            )
        elif isinstance(bt, Backtest):
            # assert scale is None, scale
            bt = replace(bt, i=i)
        else:
            if not isinstance(bt, int):
                scale = BtScale(
                    bt.index, bt.method
                )
            bt = Backtest(
                scale=scale,
                signal=signal,
                rx=rx,
                rebalance=rebalance,
                calendar=calendar,
                weight=weight,
                cap=cap,
                unit=unit,
                fx=fx,
                index=i,
                key=key,
            )
        bt.validate()
        return replace(self, bt=self.bt + (bt,)), res



#  ------------------

__all__ = [
    "Universe",
    "BtUvar",
    "BtMvar",
    "BtScale",
    "BtSignal",
    "BtValues",
    "BtParams",
    "BtResult",
    "TBtParams",
    "TBtResult",
    "Backtest",
]