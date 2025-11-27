
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
class HandleNewDailyEM(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyEMA(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).ewm_mean(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )
    
@dataclass(frozen=True)
class HandleNewDailyEMax(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_max(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            VALUE: res
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

import numba as nb

@nb.jit()
def calc_max(
    v: np.ndarray,
    is_null: np.ndarray,
    alpha: float,
    # TODO: assert float?
):
    i: int = 0
    n: int = len(v)
    res = np.empty(n, dtype=v.dtype)
    acc: float = 0
    mu: float = 0
    alpha_comp = 1 - alpha
    while i < n:
        if not is_null[i]:
            vv = v[i]
            if i == 0:
                acc = vv
                mu = acc
            elif vv > acc:
                acc = vv
            else:
                acc = (alpha_comp * acc) + (alpha * vv)
            res[i] = acc
        i += 1
    return res

@dataclass(frozen=True)
class HandleNewDailyEMid(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        mx = calc_max(v, is_null, alpha=obj.calc_alpha())
        mn = calc_min(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            "max": mx,
            "min": mn,
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(
                (pl.col("max") + pl.col("min")) / 2
            )
            .alias(VALUE)
        )
    
@dataclass(frozen=True)
class HandleNewDailyEMin(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_min(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            VALUE: res
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

import numba as nb

@nb.jit()
def calc_min(
    v: np.ndarray,
    is_null: np.ndarray,
    alpha: float,
    # TODO: assert float?
):
    i: int = 0
    n: int = len(v)
    res = np.empty(n, dtype=v.dtype)
    acc: float = 0
    alpha_comp = 1 - alpha
    while i < n:
        if not is_null[i]:
            vv = v[i]
            if i == 0:
                acc = vv
            elif vv < acc:
                acc = vv
            else:
                acc = (alpha_comp * acc) + (alpha * vv)
            res[i] = acc
        i += 1
    return res

@dataclass(frozen=True)
class HandleNewDailyEPLast(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_p_last(
            v,
            v,
            is_null,
            alpha=obj.calc_alpha(),
            l=100,
            norm = False, # TODO: parametrise in path?
        )

        return pl.DataFrame({
            "OLD": s,
            VALUE: res
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
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
    alpha: float,
    l: int,
    norm: bool,
):
    n: int = len(v)
    acc: float = 0
    res = np.zeros(n, dtype=v.dtype)
    ws = np.zeros(l, dtype=v.dtype)

    j = 0

    alpha_l = alpha ** l
    while j < l:
        ws[j] = (
            (1 - alpha) / (1 - alpha_l)
        ) * (alpha ** j)
        j += 1

    w_sum = np.sum(ws)

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
                    w = ws[k]
                    acc += vv * w
                    w_acc += w
                    k += 1
                j += 1
            
            if w_acc > 0:
                scale = w_sum / w_acc
                acc *= scale

                res[i] = acc

                if acc < min_v:
                    min_v = acc
                if acc > max_v:
                    max_v = acc

        i += 1

    return res
    
@dataclass(frozen=True)
class HandleNewDailyERms(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).pow(2).ewm_mean(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            ).sqrt()
        )


@dataclass(frozen=True)
class HandleNewDailyERng(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        mx = calc_max(v, is_null, alpha=obj.calc_alpha())
        mn = calc_min(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            "max": mx,
            "min": mn,
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(
                pl.col("max") - pl.col("min")
            )
            .alias(VALUE)
        )

@dataclass(frozen=True)
class HandleNewDailyESDev(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).ewm_std(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )

@dataclass(frozen=True)
class HandleNewDailyEVar(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedEM,
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
        obj: DailyChunkedEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).ewm_var(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )

__all__ = [
    "HandleNewDailyEM",
    "HandleNewDailyEMA",
    "HandleNewDailyEMax",
    "calc_max",
    "HandleNewDailyEMid",
    "HandleNewDailyEMin",
    "calc_min",
    "HandleNewDailyEPLast",
    "calc_p_last",
    "HandleNewDailyERms",
    "HandleNewDailyERng",
    "HandleNewDailyESDev",
    "HandleNewDailyEVar",
]