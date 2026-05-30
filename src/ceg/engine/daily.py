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

from .. import apis
from ..apis import t
from ..apis import kv

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *
from .storage import *


#  ------------------


@dataclass(frozen=True)
class Daily(Object):

    @dataclass(frozen=True)
    class Snap(Object.Snap):
        first: dt.date | None = None  # most recently added
        last: dt.date | None = None

        def with_dates(self, first: dt.date, last: dt.date):
            return replace(self, first=first, last=last)
        
        def update_from_dates(self, first: dt.date, last: dt.date):
            return replace(
                self,
                first=first,
                last=last,
            )
        def update_from_df(self, df: pl.DataFrame):
            d_min, d_max = df.pipe(
                apis.pl.select_items,
                (
                    pl.col("date").min(),
                    pl.col("date").max(),
                ),
                t=dt.date,
            )
            return replace(
                self,
                first=d_min,
                last=d_max,
            )

    @dataclass(frozen=True)
    class Params(Object.Params):
        pass

    parent: Object | str
    params: Params

    def new_snap(self) -> Daily.Snap:
        return Daily.Snap(self.id)
    
    def get_df(
        self,
        alias: str | None = None,
        scale: float = 1,
        method: str | None = None,
        with_objs: list[Daily] | None = None,
    ):
        if with_objs is not None:
            return daily_pl_df(
                [self] + with_objs,
                alias=alias,
                scale=scale,
                method=method,
            )
        return daily_pl_df(
            self,
            alias=alias,
            scale=scale,
            method=method,
        )


@dataclass(frozen=True)
class Date(DependsGraphSync, Daily):

    def self(self) -> Date:
        return self
    
    def encode_type(self, params: Daily.Params):
        return ""
    
    def encode_params(self, params: Daily.Params):
        return ""

    def handle(self, event: Event, snap: Snap):
        snap = apis.assert_is(snap, Daily.Snap)
        if not isinstance(event, GraphSync):
            return
        today = dt.date.today()
        if snap.last is None or today > snap.last:
            return ValueUpdate.if_snap_changed(
                snap,
                replace(snap, last=today)
            )
        return

date = new(Date, Date.Params)

@dataclass(frozen=True)
class DailyChunked(
    Daily,
    UseRootChunkedStorage,
):
    pass


@overload
def daily_pl_value(
    obj: Daily,
    t: Literal["df"],
    alias: str | None = None,
    scale: float = 1.0,
    expr: pl.Expr | None = None,
) -> pl.DataFrame: ...


@overload
def daily_pl_value(
    obj: Daily,
    t: Literal["series"],
    alias: str | None = None,
    scale: float = 1.0,
    expr: pl.Expr | None = None,
) -> ValuePlSeries: ...


def daily_pl_value(
    obj: Daily,
    t: Literal["df"] | Literal["series"],
    alias: str | None = None,
    scale: float = 1.0,
    expr: pl.Expr | None = None,
) -> pl.DataFrame | ValuePlSeries:
    snap = get_snap(obj)
    assert snap is not None, obj
    assert snap.first is not None, (obj, snap)
    assert snap.last is not None, (obj, snap)
    # v = obj.value_key_val().get(
    #     obj.id,
    #     t=(ValuePlSeries, ValuePlDataFrame)
    # )
    if t == "series":
        v = obj.value_key_val().get(obj.id, t=ValuePlSeries)
    elif t == "df":
        v = obj.value_key_val().get(obj.id, t=ValuePlDataFrame)
    else:
        raise ValueError(obj, t)
    assert v is not None, obj
    alias = alias or obj.id
    if expr is None:
        try:
            df = v.to_df(
                pipe=lambda e: e * scale,
                alias=alias,
            )
        except Exception as e:
            raise ValueError((v, scale, alias)) from e
    else:
        df = pl.select(
            expr.alias(alias) * scale,
            pl.date_range(snap.first, snap.last).alias(
                "date"
            ),
        )
    if t == "df":
        if expr is not None:
            return df
        if "date" in df.columns:
            return df
        return df.with_columns(
            pl.date_range(snap.first, snap.last).alias(
                "date"
            )
        )
    return ValuePlSeries.synthetic(
        df.get_column(alias),
        start=snap.first,
        end=snap.last,
    )


@overload
def daily_pl_values(
    t: Literal["df"],
    *inds: Daily | int | tuple[int, ...] | str,
    agg: str | None = None,
    scale: float = 1.0,
    alias: str = "res",
    objs: tuple[Object | str, ...] | None = None,
) -> pl.DataFrame: ...



@overload
def daily_pl_values(
    t: Literal["series"],
    *inds: Daily | int | tuple[int, ...] | str,
    agg: str,
    scale: float = 1.0,
    alias: str = "res",
    objs: tuple[Object | str, ...] | None = None,
) -> ValuePlSeries: ...


@overload
def daily_pl_values(
    t: Literal["series"],
    *inds: Daily | int | tuple[int, ...] | str,
    agg: None = None,
    scale: float = 1.0,
    alias: str = "res",
    objs: tuple[Object | str, ...] | None = None,
) -> list[ValuePlSeries]: ...

def daily_pl_values(
    t: Literal["df"] | Literal["series"],
    *inds: Daily | int | tuple[int, ...] | str,
    scale: float = 1.0,
    alias: str = "res",
    agg: str | None = None,
    objs: tuple[Object | str, ...] | None = None,
):
    i_objs = []

    for i in inds:
        if isinstance(i, int):
            i = (i,)
        if isinstance(i, Daily):
            i_objs.append(i)
        elif isinstance(i, str):
            i_objs.append(context.get().get(i, Daily))
        elif isinstance(i, tuple):
            assert objs is not None, (objs, inds)
            for ix in i:
                i_objs.append(
                    context.get().get(objs[ix], Daily)
                )
        else:
            raise ValueError(inds)

    dfs = [
        daily_pl_value(
            obj,
            "df",
        )
        for obj in i_objs
    ]

    df = apis.pl.join_dfs(
        dfs,
        on="date",
        coalesce=True,
        how="full",
    ).sort(by="date")

    (first, last) = df.pipe(
        apis.pl.select_min_max, "date", dt.date
    )

    if agg == "sum":
        df = df.select(
            "date",
            pl.sum_horizontal(pl.exclude("date")).alias(
                alias
            ),
        )
    elif agg == "mean":
        df = df.select(
            "date",
            pl.mean_horizontal(pl.exclude("date")).alias(
                alias
            ),
        )
    elif agg == "prod":
        try:
            df = df.select(
                "date",
                pl.fold(
                    acc=pl.lit(1),
                    function=lambda acc, x: acc * x,
                    exprs=pl.exclude("date"),
                    return_dtype=pl.Float64,
                ).alias(alias),
            )
        except:
            raise ValueError((objs, df))
    elif len(dfs) == 1:
        df = df.select(
            "date", pl.col(i_objs[0].id).alias(alias)
        )
    else:
        pass

    if t == "df":
        return df

    if agg is None:
        return [
            ValuePlSeries.synthetic(
                df.get_column(obj.id) * scale,
                start=first,
                end=last,
            )
            for obj in i_objs
        ]

    return ValuePlSeries.synthetic(
        df.get_column(alias) * scale, start=first, end=last
    )


def daily_pl_df(
    obj: Daily | list[Daily],
    alias: str | None = None,
    scale: float = 1.0,
    method: str | None = None,
):
    if isinstance(obj, Daily):
        return daily_pl_value(
            obj,
            "df",
            alias=alias,
            scale=scale,
        )
    alias = alias or "res"
    return daily_pl_values(
        "df",
        *obj,
        alias=alias,
        scale=scale,
        agg=method,
    )

from functools import lru_cache

@lru_cache(maxsize=32)
def daily_pl_df_cached(
    obj: Daily,
    alias: str | None = None,
    scale: float = 1.0,
    method: str | None = None,
):
    return daily_pl_df(obj, alias, scale, method)

def daily_pl_series(
    obj: Daily | list[Daily],
    alias: str | None = None,
    scale: float = 1.0,
    method: str | None = None,
):
    if isinstance(obj, Daily):
        return daily_pl_value(
            obj,
            "series",
            alias=alias,
            scale=scale,
        )
    alias = alias or "res"
    return daily_pl_values(
        "series",
        *obj,
        alias=alias,
        scale=scale,
        agg=method,
    )


@dataclass(frozen=True)
class TriageUpstreamDaily(Daily, TriageUpstream):

    @dataclass(frozen=True)
    class Snap(Daily.Snap, TriageUpstream.Snap):
        pass

__all__ = [
    "Daily",
    "TriageUpstreamDaily",
    "DailyChunked",
    "daily_pl_df",
    "daily_pl_df_cached",
    "daily_pl_series",
    "daily_pl_value",
    "daily_pl_values",
    "Date",
    "date",
]
