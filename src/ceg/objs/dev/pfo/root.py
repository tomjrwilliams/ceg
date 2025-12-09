
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
    method: str | None = None
    scale: float = 1.
    activation: str | None = None

@dataclass(frozen=True)
class PfolioSignal():
    index: tuple[int, ...]
    method: str | None = None
    scale: float = 1.
    activation: str | None = None

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

def apply_activation(
    ser: pl.Series,
    activation: str | None = None
):
    if activation is None:
        return ser.fill_nan(None)
    activation = (
        activation.replace("ng", "neg")
        .replace("tnlog", "tanhlog")
    )
    if activation == "tanh":
        return ser.tanh().fill_nan(None)
    if activation == "neg":
        return (-1 * ser).fill_nan(None)
    elif activation == "neg_tanhlog":
        return (
            -1 * ser.log().tanh()
        ).fill_nan(None)
    elif activation == "neg_tanhlog_neg":
        return (
            -1 * (-1 * ser).log().tanh()
        ).fill_nan(None)
    elif activation == "n_tanh_p_one":
        return (
            pl.DataFrame({VALUE: ser.tanh()})
            .select(
                pl.when(pl.col(VALUE).is_null())
                .then(None)
                .when(pl.col(VALUE) < 0)
                .then(pl.col(VALUE))
                .otherwise(1)
                .alias(VALUE)
            )
            .get_column(VALUE).fill_nan(None)
        )
    elif activation == "n_one_p_tanh":
        return (
            pl.DataFrame({VALUE: ser.tanh()})
            .select(
                pl.when(pl.col(VALUE).is_null())
                .then(None)
                .when(pl.col(VALUE) > 0)
                .then(pl.col(VALUE))
                .otherwise(-1)
                .alias(VALUE)
            )
            .get_column(VALUE).fill_nan(None)
        )
    else:
        raise ValueError((ser, activation))


def get_value(
    obj: DailyChunked | list[DailyChunked],
    domain: Domain,
    scale: float = 1.0,
    method: str | None = None,
    activation: str | None = None,
):
    if isinstance(obj, Object):
        ser = (
            obj.value_key_val(domain)
            .get(obj.path, kv.BODY, ValuePlSeries)
            .ser
        )
        ser = apply_activation(ser, activation) * scale
        snap = obj.snap_key_val(domain).get(
            obj.path, DailySnap
        )
        assert snap is not None, obj
        return ValuePlSeries.synthetic(
            ser, start=snap.first, end=snap.last
        )
    vs = [
        o.value_key_val(domain)
        .get(o.path, kv.BODY, ValuePlSeries)
        for o in obj
    ]
    snap = obj[0].snap_key_val(domain).get(
        obj[0].path, DailySnap
    )
    assert snap is not None, obj

    # TODO: df join if given diff dates by snap

    df = pl.DataFrame([v.ser.alias(v.obj) for v in vs])
    if method == "sum":
        df = df.select(
            pl.sum_horizontal(pl.all()).alias("res")
        )
    elif method == "mean":
        df = df.select(
            pl.mean_horizontal(pl.all()).alias("res")
        )
    elif method == "prod":
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
        assert len(obj) == 1, obj
        df = df.select(
            pl.col(vs[0].obj).alias("res")
        )
    ser = df.get_column("res")
    ser = apply_activation(ser, activation) * scale
    return ValuePlSeries.synthetic(
        ser, start=snap.first, end=snap.last
    )

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
            return get_value(
                obj,
                domain,
            )
        objs = [
            self.get_scale(ii, domain, DailyChunked)
            for ii in i.index
        ]
        return get_value(
            objs,
            domain,
            method=i.method,
            scale=i.scale,
            activation=i.activation,
        )

    def get_signal(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.signal is None:
            return None
        return domain.get(self.signal[i], t=t)
    
    def signal_value(
        self, i: int | PfolioSignal, domain: Domain, 
    ) -> ValuePlSeries:
        if isinstance(i, int):
            obj = self.get_signal(i, domain, DailyChunked)
            assert obj is not None, obj
            return get_value(
                obj,
                domain,
            )
        objs = [
            self.get_signal(ii, domain, DailyChunked)
            for ii in i.index
        ]
        assert all([obj is not None for obj in objs]), objs
        return get_value(
            objs, # type: ignore
            domain,
            method=i.method,
            scale=i.scale,
            activation=i.activation,
        )
    
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
                apis.cal.rebalance_mask(i)
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

__all__ = ["Universe", "PfolioScale", "PfolioSignal"]