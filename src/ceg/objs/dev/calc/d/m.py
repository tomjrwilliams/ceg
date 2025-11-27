
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

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

from .base import *

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyM(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyMA(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).rolling_mean(
                len(df) 
                if params.window is None 
                else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    
@dataclass(frozen=True)
class HandleNewDailyKurt(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).rolling_kurtosis(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

@dataclass(frozen=True)
class HandleNewDailyMax(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_max(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    

@dataclass(frozen=True)
class HandleNewDailyMed(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_median(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

@dataclass(frozen=True)
class DailyMidParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    width: float | None = None

@dataclass(frozen=True)
class DailyChunkedMid(DailyChunkedM):

    params: DailyMidParams

    
@dataclass(frozen=True)
class HandleNewDailyMid(HandleNewDailyM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedMid,
        domain: Domain,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
            obj,
            domain,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedMid,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
        e = (u + l) / 2

        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
        return df
    

@dataclass(frozen=True)
class HandleNewDailyMin(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        
        return(
            pl.col(VALUE).rolling_min(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )


@dataclass(frozen=True)
class HandleNewDailyPLast(HandleNewDailyM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
            obj,
            domain,
        )
    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_p_last(
            v,
            v,
            is_null,
            l=len(df) if params.window is None else params.window,
            norm = False, # TODO: parametrise in path?
        )

        return pl.DataFrame({
            "mask": s,
            VALUE: res
        }).select(
            pl.when(pl.col("mask").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

import numba as nb

# rolling window, indicator series of gt last
# then exp decay over binary series?
# so actually needs sliding window of weights

@nb.jit()
def calc_p_last(
    v: np.ndarray,
    loc: np.ndarray,
    is_null: np.ndarray,
    l: int,
    norm: bool,
):
    n: int = len(v)
    acc: float = 0
    res = np.zeros(n, dtype=v.dtype)

    l -= 1

    j = 0

    min_v = 1
    max_v = 0

    i: int = 0
    while i < n:
        if i == 0:
            acc = 0.5
            res[i] = acc
        elif not is_null[i]:
            j = 1
            k = 0

            acc = 0
            w_acc = 0

            while k < l and j <= i:
                ij = i - j
                if not is_null[ij]:
                    vv = (
                        0 
                        if v[ij] < loc[i]
                        else 0.5 if v[ij] == loc[i]
                        else 1
                    )
                    w = 1 / (l + 1)
                    acc += vv * w
                    w_acc += w
                    k += 1
                j += 1
            
            if w_acc > 0:
                scale = 1 / w_acc
                acc *= scale

                res[i] = acc

                if acc < min_v:
                    min_v = acc
                if acc > max_v:
                    max_v = acc

        i += 1
        
    return res
    
@dataclass(frozen=True)
class DailyQileParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    quantile: float | None = None

@dataclass(frozen=True)
class DailyChunkedQile(DailyChunkedM):

    params: DailyQileParams

@dataclass(frozen=True)
class HandleNewDailyQile(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedQile,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )
        
    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedQile,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        assert params.quantile is not None, obj
        
        return(
            pl.col(VALUE).rolling_quantile(
                params.quantile,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    

@dataclass(frozen=True)
class HandleNewDailyRank(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_rank(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

    
@dataclass(frozen=True)
class HandleNewDailyRms(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).pow(2).rolling_mean(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            ).sqrt()
        )

    
@dataclass(frozen=True)
class DailyMRngParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    width: float | None = None

@dataclass(frozen=True)
class DailyChunkedMRng(DailyChunkedM):

    params: DailyMRngParams


@dataclass(frozen=True)
class HandleNewDailyRng(HandleNewDailyM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedMRng,
        domain: Domain,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
            obj,
            domain,
        )
    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedMRng,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
        e = u - l

        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
        return df

@dataclass(frozen=True)
class HandleNewDailySDev(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_std(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    
@dataclass(frozen=True)
class HandleNewDailySkew(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_skew(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    

@dataclass(frozen=True)
class HandleNewDailyVar(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_var(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

#  ------------------

__all__ = [
    "HandleNewDailyM",
    "HandleNewDailyMA",
    "HandleNewDailyKurt",
    "HandleNewDailyMax",
    "HandleNewDailyMed",
    "DailyMidParams",
    "DailyChunkedMid",
    "HandleNewDailyMid",
    "HandleNewDailyMin",
    "HandleNewDailyPLast",
    "calc_p_last",
    "DailyQileParams",
    "DailyChunkedQile",
    "HandleNewDailyQile",
    "HandleNewDailyRank",
    "HandleNewDailyRms",
    "DailyMRngParams",
    "DailyChunkedMRng",
    "HandleNewDailyRng",
    "HandleNewDailySDev",
    "HandleNewDailySkew",
    "HandleNewDailyVar",
]