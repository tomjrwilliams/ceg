
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

@dataclass(frozen=True)
class PfolioScale():
    index: tuple[int, ...]
    method: str

# TODO: rename daily, move under d

def find_object(
    obj: Object | str, objs: tuple[Object | str, ...] | None
) -> int | None:
    if objs is None:
        return None
    path = obj if isinstance(obj, str) else obj.path
    for i, o in enumerate(objs):
        if isinstance(o, str) and o == path:
            return i
        elif isinstance(o, Object) and o.path == path:
            return i
    return None

@dataclass(frozen=True)
class Universe(
    UseRootConfigs,
):
    parent: EngineRoot | str

    name: str
    ident: int # config specifier (used for path, and to generate on the fly the below)

    # TODO: daily chunked not just object?

    scale: tuple[Object | str, ...]
    signal: tuple[Object | str, ...] = ()
    rebalance: tuple[Object | str, ...] = ()
    calendar: tuple[Object | str, ...] = ()
    weight: tuple[Object | str, ...] = ()
    
    # rx_label, scale_label, etc.?

    # to weight, generate a new portfolio and add weight
    
    version: int = 0
    type: Literal["Universe"] = "Universe"

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, t=EngineRoot)
    
    @classmethod
    def new(
        cls,
        parent: EngineRoot,
        name: str,
        ident: int,
        scale: tuple[Object | str, ...] = (),
        signal: tuple[Object | str, ...] = (),
        rebalance: tuple[Object | str, ...] = (),
        calendar: tuple[Object | str, ...] = (),
        weight: tuple[Object | str, ...] = (),
    ):
        return cls(
            parent=parent,
            path=f"pf.{name}.{ident}",
            name=name,
            ident=ident,
            scale=scale,
            signal=signal,
            rebalance=rebalance,
            calendar=calendar,
            weight=weight,
        )
    
    # TODO; overload to return multiple indices if more than oen passed?

    def with_scale(self, scale: Object | str) -> tuple[Universe, int]:
        i = find_object(scale, self.scale)
        if i is not None:
            return self, i
        scales = self.scale + (scale,)
        return replace(
            self, scale=scales
        ), len(scales) - 1
    
    def with_signal(self, signal: Object | str) -> tuple[Universe, int]:
        i = find_object(signal, self.signal)
        if i is not None:
            return self, i
        signals = (
            () if self.signal is None else self.signal
        ) + (signal,)
        return replace(
            self, signal=signals
        ), len(signals) - 1
    
    def with_rebalance(self, rebalance: Object | str) -> tuple[Universe, int]:
        i = find_object(rebalance, self.rebalance)
        if i is not None:
            return self, i
        rebalances = (
            () if self.rebalance is None else self.rebalance
        ) + (rebalance,)
        return replace(
            self, rebalance=rebalances
        ), len(rebalances) - 1
    
    def with_calendar(self, calendar: Object | str) -> tuple[Universe, int]:
        i = find_object(calendar, self.calendar)
        if i is not None:
            return self, i
        calendars = (
            () if self.calendar is None else self.calendar
        ) + (calendar,)
        return replace(
            self, calendar=calendars
        ), len(calendars) - 1
    
    def with_weight(self, weight: Object | str) -> tuple[Universe, int]:
        i = find_object(weight, self.weight)
        if i is not None:
            return self, i
        weights = (
            () if self.weight is None else self.weight
        ) + (weight,)
        return replace(
            self, weight=weights
        ), len(weights) - 1
    
    def get_scale(
        self, i: int, domain: Domain, t: Type[O]
    ) -> O:
        return domain.get(self.scale[i], t=t)
    
    def scale_value(
        self, i: int | PfolioScale, domain: Domain, 
    ) -> ValuePlSeries:
        if isinstance(i, int):
            obj = self.get_scale(i, domain, DailyChunked)
            v = (
                obj.value_key_val(domain)
                .get(obj.path, kv.BODY, ValuePlSeries)
            )
            return v
        objs = [
            self.get_scale(ii, domain, DailyChunked)
            for ii in i.index
        ]
        vs = [
            o.value_key_val(domain)
            .get(o.path, kv.BODY, ValuePlSeries)
            for o in objs
        ]
        df = pl.DataFrame([v.ser.alias(v.obj) for v in vs])
        if i.method == "sum":
            df = df.select(
                pl.sum_horizontal(pl.all()).alias("res")
            )
        elif i.method == "mean":
            df = df.select(
                pl.mean_horizontal(pl.all()).alias("res")
            )
        elif i.method == "prod":
            df = df.select(
                pl.fold(
                    acc=pl.lit(1), 
                    function=lambda acc, x: acc * x, 
                    exprs=pl.all(),
                    return_dtype=pl.Float64,
                )
                .alias("res")
            )
        else:
            raise ValueError(i)
        ser = df.get_column("res")
        return ValuePlSeries.synthetic(ser)

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
    
    def rebalance_value(
        self, 
        i: int | str | None, 
        domain: Domain, 
        value: DailyChunked | None = None,
    ):
        if i is None:
            return None
        elif isinstance(i, int):
            obj = self.get_rebalance(i, domain, DailyChunked)
            assert obj is not None, i
            v = (
                obj.value_key_val(domain)
                .get(obj.path, kv.BODY, ValuePlSeries)
            )
            return v
        else:
            assert value is not None, value
            snap = value.get_snap(domain)
            assert isinstance(snap, DailySnap), snap
            df = pl.select(
                pl.date_range(snap.first, snap.last)
                .alias("date")
            ).select(
                rebalance_mask(i)
                .alias("mask")
            )
            return ValuePlSeries.synthetic(
                df.get_column("mask")
            )

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

def rebalance_mask(s: str):
    # TODO: range selection, set selection
    # TODO: every n (second month, etc.)

    d = pl.col("date")
    w = d.dt.week()
    m = d.dt.month()
    y = d.dt.year()
    m_start = d.dt.month_start()
    m_weekday = m_start.dt.weekday()
    w_in_m = (
        (d - m_start).dt.total_days() / 7
    ).floor() + 1
    exprs = []
    n_d = n_w = n_m = None

    if "D" in s:
        n_d, s = s.split("D")
        n_d = int(n_d)
        # TODO: many?
        exprs.append(d.dt.weekday() == n_d)
    if "W" in s:
        n_w, s = s.split("W")
        n_w = int(n_w)
        exprs.append(w_in_m == n_w)
    if "M" in s:
        n_m, s = s.split("M")
        n_m = int(n_m)
        exprs.append(m == n_m)
    return pl.all_horizontal(*exprs)

__all__ = ["Universe", "PfolioScale"]