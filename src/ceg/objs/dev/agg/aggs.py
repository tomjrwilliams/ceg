from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
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
from functools import partial

from frozendict import frozendict

import numpy as np
import polars as pl
import numba as nb

from .... import apis
from ....apis import assert_is, field_default, t, kv, logs
from ....engine import *

from ..calc import *

from .containers import *
from .composition import (
    DailyComposition,
    DailyOutput,
    HandleNewDailyComposition,
    get_composition_dfs,
)

from ....apis.calcs.regimes.dd import *

from ....apis.calcs import *

logger, log = logs.logger(
    __file__, streams=dict(formatter=True)
)

#  ------------------

# DailyAggregate / DailyAggregateDeriv / get_agg_dfs / HandleNewDailyAggregate
# have been replaced by DailyComposition / DailyOutput / get_composition_dfs /
# HandleNewDailyComposition in .composition. Handler subclasses below inherit
# from HandleNewDailyComposition and override `agg_dfs()` with the concrete
# calculation only.


@dataclass(frozen=True)
class HandleNewDailyHSum(HandleNewDailyComposition):
    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        return df.select(
            pl.col("date"),
            h_sum(
                [
                    k
                    for k in df.schema.keys()
                    if k != "date"
                ],
                propagate_nulls=False,
            )
            .alias(VALUE)
            .fill_nan(None),
        )


@dataclass(frozen=True)
class HandleNewDailyHGross(HandleNewDailyComposition):
    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        df = df.select(
            pl.col("date"),
            h_sum(
                [
                    k
                    for k in df.schema.keys()
                    if k != "date"
                ],
                f=lambda e: e.abs().fill_null(0.0),
                propagate_nulls=False,
            )
            .alias(VALUE)
            .fill_nan(None),
        )
        return df


@dataclass(frozen=True)
class HandleNewDailyHProd(HandleNewDailyComposition):
    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        df = df.select(
            pl.col("date"),
            h_prod(
                [
                    k
                    for k in df.schema.keys()
                    if k != "date"
                ],
                propagate_nulls=True,
            )
            .alias(VALUE)
            .fill_nan(None),
        )
        return df


def h_sum(
    ks: list[str],
    propagate_nulls: bool = True,
    f: Callable[[pl.Expr], pl.Expr] | None = None,
):
    return (
        pl.when(
            pl.all_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
            if not propagate_nulls
            else pl.any_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
        )
        .then(None)
        .otherwise(
            pl.sum_horizontal(
                *[
                    pl.col(k)
                    if f is None
                    else pl.col(k).pipe(f)
                    for k in ks
                ]
            )
        )
    )


def h_prod(
    ks: list[str],
    propagate_nulls: bool = True,
    f: Callable[[pl.Expr], pl.Expr] | None = None,
):
    return (
        pl.when(
            pl.all_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
            if not propagate_nulls
            else pl.any_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
        )
        .then(None)
        .otherwise(
            pl.fold(
                acc=pl.lit(1.0),
                function=lambda acc, x: acc * x,
                exprs=[
                    pl.col(k).fill_nan(None)
                    if f is None
                    else pl.col(k).fill_nan(None).pipe(f)
                    for k in ks
                ],
            )
        )
    )


@dataclass(frozen=True)
class HandleNewDailyHDot(HandleNewDailyComposition):
    dot_groups: int

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        ks = list(df.drop("date").schema.keys())
        n = self.dot_groups
        d = len(ks) / self.dot_groups
        assert d == int(d), self
        groups = [
            [ks[int((ii * d) + i)] for ii in range(n)]
            for i in range(int(d))
        ]
        df = df.select(
            pl.col("date"),
            *[
                h_prod(
                    g,
                    propagate_nulls=False,
                    f=lambda e: e.fill_null(0.0),
                )
                .alias(f"{VALUE}_{i}")
                .fill_nan(None)
                for i, g in enumerate(groups)
            ],
        ).select(
            "date",
            h_sum(
                [f"{VALUE}_{i}" for i in range(int(d))],
                propagate_nulls=True,
            )
            .alias(VALUE)
            .fill_nan(None),
        )
        return df


@dataclass(frozen=True)
class HandleNewDailyDDRegime(HandleNewDailyComposition):
    threshold: float
    span: int
    window: int
    continuation: float
    share_dd: bool
    share_recovery: bool
    continuous: bool

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return dd_regime_df(
            rxs,
            threshold=self.threshold,
            span=self.span,
            window=self.window,
            continuation=self.continuation,
            share_dd=self.share_dd,
            share_recovery=self.share_recovery,
            continuous=self.continuous,
        )


@dataclass(frozen=True)
class HandleNewDailyMetricsEWM(HandleNewDailyComposition):
    span: float

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return metrics_ewm_df(
            rxs,
            span=self.span,
        )


@dataclass(frozen=True)
class HandleNewDailyMetricsGrid(HandleNewDailyComposition):
    rx_grid_width: float
    re_grid: bool

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return metrics_grid_df(
            rxs,
            rx_grid_width=self.rx_grid_width,
            re_grid=self.re_grid,
        )


@dataclass(frozen=True)
class HandleNewDailyWeightsRank(HandleNewDailyComposition):
    metric: str
    sample: str | int
    window: float
    smooth: float
    norm: str
    top: int

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return weights_rank_df(
            rxs,
            metric=self.metric,
            sample=self.sample,
            window=self.window,
            smoothing=self.smooth,
            norm=self.norm,
            top=self.top,
        )


HANDLERS = {
    "hsum": HandleNewDailyHSum,
    "hdot": HandleNewDailyHDot,
    "hprod": HandleNewDailyHProd,
    "hgross": HandleNewDailyHGross,
    "dd_regime": HandleNewDailyDDRegime,
    "metrics_ewm": HandleNewDailyMetricsEWM,
    "metrics_grid": HandleNewDailyMetricsGrid,
    "weights_rank": HandleNewDailyWeightsRank,
}


def agg_handler(
    agg: str,
    params: frozendict[str, bool | int | float | None]
    | None = None,
) -> HandleNewDailyComposition:
    cls = assert_is(HANDLERS.get(agg))
    if params is None:
        return cls()
    try:
        return cls(**params)
    except:
        raise ValueError((cls, agg, params))


__all__ = [
    "agg_handler",
    "HANDLERS",
]
