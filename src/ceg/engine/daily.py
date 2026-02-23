
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

# @dataclass(frozen=True)
# class DailyTags:
#     pass

@dataclass(frozen=True)
class DailyParams:
    pass

@dataclass(frozen=True)
class Daily(Object):
    parent: Object | str
    # tags: DailyTags
    # params: DailyParams

    # version: int = 0
    # type: Literal["DailyM"] = "DailyM"
    # NOTE: to change

    @dataclass(frozen=True) 
    class Snap(Snap):
        first: dt.date | None = None # most recently added
        last: dt.date | None = None

    @overload
    def get_snap(
        self,
        t: None = None,
    ) -> Daily.Snap: ...
        
    @overload
    def get_snap(
        self,
        t: Type[TDailySnap],
    ) -> TDailySnap: ...
        
    def get_snap(
        self,
        t: Type[Daily.Snap] | None = None,
    ) -> Daily.Snap:
        s = self._get_snap()
        if s is None:
            s = type(self).Snap.new(self.path)
        if t is None:
            t = Daily.Snap
        assert isinstance(s, t), (s, t)
        return s


TDailySnap = TypeVar("TDailySnap", bound=Daily.Snap)

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
    snap = obj.get_snap(t = Daily.Snap)
    assert snap is not None, (obj)
    assert snap.first is not None, (obj, snap)
    assert snap.last is not None, (obj, snap)
    v = (
        obj.value_key_val()
        .get(obj.path, t=ValuePlSeries)
    )
    assert v is not None, obj
    alias = alias or obj.path
    if expr is None:
        try:
            df = v.to_df(
                pipe = lambda e: e * scale,
                alias=alias,
            )
        except:
            raise ValueError((v, scale, alias))
    else:
        df = pl.select(
            expr.alias(alias) * scale,
            pl.date_range(snap.first, snap.last)
            .alias("date")
        )
    if t == "df":
        if expr is not None:
            return df
        return df.with_columns(
            pl.date_range(snap.first, snap.last)
            .alias("date")
        )
    return ValuePlSeries.synthetic(
        df.get_column(alias),
        start=snap.first, 
        end=snap.last
    )

@overload
def daily_pl_values(
    t: Literal["df"],
    *inds: Daily | int | tuple[int, ...] | str,
    scale: float = 1.0,
    alias: str = "res",
    agg: str | None = None,
    objs: tuple[Object | str, ...] | None = None,
) -> pl.DataFrame: ...

@overload
def daily_pl_values(
    t: Literal["series"],
    *inds: Daily | int | tuple[int, ...] | str,
    scale: float = 1.0,
    alias: str = "res",
    agg: None = None,
    objs: tuple[Object | str, ...] | None = None,
) -> list[ValuePlSeries]: ...

@overload
def daily_pl_values(
    t: Literal["series"],
    *inds: Daily | int | tuple[int, ...] | str,
    scale: float = 1.0,
    alias: str = "res",
    agg: str = "",
    objs: tuple[Object | str, ...] | None = None,
) -> ValuePlSeries: ...

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
                i_objs.append(context.get().get(
                    objs[ix], Daily
                ))
        else:
            raise ValueError(inds)
        
    dfs = [
        daily_pl_value(
            obj,"df", 
        )
        for obj in i_objs
    ]

    df = apis.pl.join_dfs(
        dfs,
        on="date",
        coalesce=True,
        how="full",
    ).sort(by="date")

    (
        first, last
    ) = df.pipe(apis.pl.select_min_max, "date", dt.date)

    if agg == "sum":
        df = df.select(
            "date",
            pl.sum_horizontal(pl.exclude("date"))
            .alias(alias)
        )
    elif agg == "mean":
        df = df.select(
            "date",
            pl.mean_horizontal(pl.exclude("date"))
            .alias(alias)
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
                )
                .alias(alias)
            )
        except:
            raise ValueError((objs, df))
    elif len(dfs) == 1:
        df = df.select(
            "date",
            pl.col(i_objs[0].path).alias(alias)
        )
    else:
        pass
    
    if t == "df":
        return df
    
    if agg is None:
        return [
            ValuePlSeries.synthetic(
                df.get_column(obj.path) * scale,
                start=first,
                end=last
            )
            for obj in i_objs
        ]
    
    return ValuePlSeries.synthetic(
        df.get_column(alias) * scale,
        start=first,
        end=last
    )


def daily_pl_df(
    obj: Daily | list[Daily],
    alias: str | None = None,
    scale: float = 1.0,
    method: str | None = None,
):
    if isinstance(obj, Daily):
        return daily_pl_value(
            obj,"df", alias=alias, scale=scale,
        )
    alias = alias or "res"
    return daily_pl_values(
        "df",
        *obj,
        alias=alias,
        scale=scale,
        agg=method,
    )

def daily_pl_series(
    obj: Daily | list[Daily],
    alias: str | None = None,
    scale: float = 1.0,
    method: str | None = None,
):
    if isinstance(obj, Daily):
        return daily_pl_value(
            obj,"series", alias=alias, scale=scale,
        )
    alias = alias or "res"
    return daily_pl_values(
        "series",
        *obj,
        alias=alias,
        scale=scale,
        agg=method,
    )

__all__ = [
    "DailyParams",
    "Daily",
    "DailyChunked",
    "daily_pl_df",
    "daily_pl_series",
    "daily_pl_value",
    "daily_pl_values",
]