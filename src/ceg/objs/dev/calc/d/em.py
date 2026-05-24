from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    ParamSpec,
    Callable,
    Concatenate,
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
import math

from ..... import apis
from .....apis import assert_is, t, kv, logs
from .....apis.pl import *

from .....engine import *
from ...bars.d import *

from .....apis.calcs.utils import *

P = ParamSpec("P")
O = TypeVar("O", bound=Object)

logger, log = logs.logger(
    __file__, streams=dict(formatter=True)
)

#  ------------------


def calc_alpha(span):
    return 2 / (span + 1)


@dataclass(frozen=True)
class DailyEM(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    SUFFIX: ClassVar[str] = "EM"

    @dataclass(frozen=True)
    class Params(Daily.Params):
        span: float | None = None
        alpha: float | None = None

        ann: bool = False

        cum: bool = False

        def calc_alpha(self) -> float:
            res = (
                self.alpha
                if self.alpha is not None
                else calc_alpha(self.span)
                if self.span is not None
                else None
            )
            assert res is not None, self
            return res
        
        def encode_ewm(self):
            tags = []
            if self.span is None:
                alpha = assert_is(self.alpha, float)
                tags.append(alpha)
            elif self.alpha is None:
                span = assert_is(self.span, float)
                tags.append(span)
            else:
                raise ValueError(self)
            if self.ann:
                tags.append("ann")
            if self.cum:
                tags.append("cum")
            return "-".join(map(str, tags))

    parent: str | Daily
    params: Params

    def self(self) -> DailyEM:
        return self
    
    def get_parent(self) -> Daily:
        return get_parent(self)

    # def encode_type(self, params: DailyEM.Params):
    #     return self.SUFFIX

    # def encode_params(self, params: DailyEM.Params):
    #     return params.encode_ewm()

    def calc_expr(self, df: pl.DataFrame, event: ValueUpdate):
        raise NotImplementedError(self)
    
    def calc_df(self, df: pl.DataFrame, event: ValueUpdate):
        raise NotImplementedError(self)

    def handle(self, event: Event, snap: Snap):
        assert self.SUFFIX != "EM", self
        
        event = assert_is(event, ValueUpdate)
        snap = assert_is(snap, self.Snap)

        df = self.get_parent().get_df(alias=VALUE)

        try:
            res = self.calc_df(df, event)
        except NotImplementedError:
            e = self.calc_expr(df, event)
            res = df.with_columns(e)
            
        _ = self.set_value_from_df(
            res,
            VALUE,
        )
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(res),
        )

#  ------------------


@dataclass(frozen=True)
class DailyEMA(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEMA.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ema-4' [f64]
    [
        5711.550199
        null
        null
        5753.13012
        5782.478072
        5773.286843
        5759.672106
        5705.003264
    ]
    """

    SUFFIX: ClassVar[str] = "ema"

    def self(self) -> DailyEMA:
        return self

    def encode_type(self, params: DailyEMA.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEMA.Params):
        return params.encode_ewm()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        params = self.params
        return pl.col(VALUE).ewm_mean(
            span=params.span,
            alpha=params.alpha,
            min_samples=1,  # TODO: or more?
            ignore_nulls=True,
        )

daily_ema = new(DailyEMA, DailyEMA.Params)

@dataclass(frozen=True)
class DailyEMax(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEMax.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emax-4' [f64]
    [
        5721.07
        null
        null
        5815.5
        5826.5
        5799.7
        5775.52
        5714.512
    ]
    """

    SUFFIX: ClassVar[str] = "emax"

    def self(self) -> DailyEMax:
        return self

    def encode_type(self, params: DailyEMax.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEMax.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_max(v, is_null, alpha=params.calc_alpha())
        return pl.DataFrame(
            {"OLD": s, VALUE: res}
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )
    
daily_emax = new(DailyEMax, DailyEMax.Params)


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
class DailyEMid(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEMid.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emid-4' [f64]
    [
        5714.536
        null
        null
        5783.2506
        5803.85036
        5779.6
        5757.385
        5668.756
    ]
    """

    SUFFIX: ClassVar[str] = "emid"

    def self(self) -> DailyEMid:
        return self

    def encode_type(self, params: DailyEMid.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEMid.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        mx = calc_max(v, is_null, alpha=params.calc_alpha())
        mn = calc_min(v, is_null, alpha=params.calc_alpha())
        return pl.DataFrame(
            {
                "OLD": s,
                "max": mx,
                "min": mn,
            }
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise((pl.col("max") + pl.col("min")) / 2)
            .alias(VALUE)
        )

daily_emid = new(DailyEMid, DailyEMid.Params)

@dataclass(frozen=True)
class DailyEMin(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEMin.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emin-4' [f64]
    [
        5708.002
        null
        null
        5751.0012
        5781.20072
        5759.5
        5739.25
        5623.0
    ]
    """
    SUFFIX: ClassVar[str] = "emin"

    def self(self) -> DailyEMin:
        return self

    def encode_type(self, params: DailyEMin.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEMin.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_min(v, is_null, alpha=params.calc_alpha())
        return pl.DataFrame(
            {"OLD": s, VALUE: res}
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

daily_emin = new(DailyEMin, DailyEMin.Params)

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
class DailyELwm(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyELwm.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-elwm-4' [f64]
    [
        5721.07
        null
        null
        5815.5
        5826.5
        5799.7
        5775.52
        5714.512
    ]
    """

    SUFFIX: ClassVar[str] = "elwm"

    def self(self) -> DailyELwm:
        return self

    def encode_type(self, params: DailyELwm.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyELwm.Params):
        return params.encode_ewm()

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        is_null = s.is_null().to_numpy()
        if params.cum:
            s = s.fill_nan(None).cum_sum()
        v = s.fill_null(0).to_numpy()
        res = calc_lwm(v, is_null, alpha=params.calc_alpha())
        return pl.DataFrame(
            {"OLD": s, VALUE: res}
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

daily_elwm = new(DailyELwm, DailyELwm.Params)

@dataclass(frozen=True)
class DailyEHwm(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEHwm.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ehwm-4' [f64]
    [
        5721.07
        null
        null
        5815.5
        5826.5
        5799.7
        5775.52
        5714.512
    ]
    """

    SUFFIX: ClassVar[str] = "ehwm"

    def self(self) -> DailyEHwm:
        return self

    def encode_type(self, params: DailyEHwm.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEHwm.Params):
        return params.encode_ewm()

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        is_null = s.is_null().to_numpy()
        if params.cum:
            s = s.fill_nan(None).cum_sum()
        v = s.fill_null(0).to_numpy()
        res = calc_hwm(v, is_null, alpha=params.calc_alpha())
        return pl.DataFrame(
            {"OLD": s, VALUE: res}
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

daily_ehwm = new(DailyEHwm, DailyEHwm.Params)

import numba as nb


@nb.jit()
def calc_lwm(
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
    lwm: float = 0.0
    while i < n:
        if not is_null[i]:
            vv = v[i]
            if i == 0 or vv < acc:
                acc = vv
                lwm = acc
            else:
                acc = (alpha_comp * acc) + (alpha * vv)
            res[i] = lwm
        i += 1
    return res


@dataclass(frozen=True)
class DailyEPLast(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEPLast.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eplast-4' [f64]
    [
        0.278662
        null
        null
        0.00008
        0.000032
        0.840017
        0.936007
        0.999937
    ]
    """

    SUFFIX: ClassVar[str] = "eplast"

    def self(self) -> DailyEPLast:
        return self

    def encode_type(self, params: DailyEPLast.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEPLast.Params):
        return params.encode_ewm()

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_p_last(
            v,
            v,
            is_null,
            alpha=params.calc_alpha(),
            l=100,
            norm=False,  # TODO: parametrise in path?
        )

        return pl.DataFrame(
            {"OLD": s, VALUE: res}
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

daily_eplast = new(DailyEPLast, DailyEPLast.Params)

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

    alpha_l = alpha**l
    while j < l:
        ws[j] = ((1 - alpha) / (1 - alpha_l)) * (alpha**j)
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
                        else 0.5
                        if v[ij] == loc[i]
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
class DailyERms(DailyEM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyERms.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-erms-4' [f64]
    [
        0.007089
        null
        null
        0.012077
        0.009431
        0.010308
        0.008288
        0.014329
    ]
    """

    SUFFIX: ClassVar[str] = "erms"

    def self(self) -> DailyERms:
        return self

    def encode_type(self, params: DailyERms.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyERms.Params):
        return params.encode_ewm()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        params = self.params
        ann = 1 if not params.ann else 16
        min_samples=max(
            [8, int(assert_is(params.span) / 64)]
        )
        i_range = pl.int_range(0, len(df))
        e = pl.col(VALUE)
        e_global = e.pow(2).mean().sqrt().mul(ann)
        e_warm = pl.when(i_range < min_samples).then(e_global).otherwise(e)
        return (
            e_warm.pow(2)
            .ewm_mean(
                span=params.span,
                alpha=params.alpha,
                ignore_nulls=True,
                min_samples=1,
            )
            .sqrt()
            .mul(ann)
        )

daily_erms = new(DailyERms, DailyERms.Params)

@dataclass(frozen=True)
class DailyERng(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyERng.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-erng-4' [f64]
    [
        13.068
        null
        null
        64.4988
        45.29928
        40.2
        36.27
        91.512
    ]
    """


    SUFFIX: ClassVar[str] = "erng"

    def self(self) -> DailyERng:
        return self

    def encode_type(self, params: DailyERng.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyERng.Params):
        return params.encode_ewm()

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        mx = calc_max(v, is_null, alpha=params.calc_alpha())
        mn = calc_min(v, is_null, alpha=params.calc_alpha())
        return pl.DataFrame(
            {
                "OLD": s,
                "max": mx,
                "min": mn,
            }
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(pl.col("max") - pl.col("min"))
            .alias(VALUE)
        )

daily_erng = new(DailyERng, DailyERng.Params)

@dataclass(frozen=True)
class DailyESDev(DailyEM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyESDev.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-esdev-4' [f64]
    [
        0.008131
        null
        null
        0.011117
        0.009137
        0.011772
        0.009188
        0.012394
    ]
    """

    SUFFIX: ClassVar[str] = "esdev"

    def self(self) -> DailyESDev:
        return self

    def encode_type(self, params: DailyESDev.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyESDev.Params):
        return params.encode_ewm()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        params = self.params
        return pl.col(VALUE).ewm_std(
            span=params.span,
            alpha=params.alpha,
            min_samples=1,  # TODO: or more?
            ignore_nulls=True,
        )
    
daily_esdev = new(DailyESDev, DailyESDev.Params)


@dataclass(frozen=True)
class DailyEVar(DailyEM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEVar.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-evar-4' [f64]
    [
        0.000066
        null
        null
        0.000124
        0.000083
        0.000139
        0.000084
        0.000154
    ]
    """

    SUFFIX: ClassVar[str] = "evar"

    def self(self) -> DailyEVar:
        return self

    def encode_type(self, params: DailyEVar.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEVar.Params):
        return params.encode_ewm()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        params = self.params
        return pl.col(VALUE).ewm_var(
            span=params.span,
            alpha=params.alpha,
            min_samples=1,  # TODO: or more?
            ignore_nulls=True,
        )

daily_evar = new(DailyEVar, DailyEVar.Params)

@dataclass(frozen=True)
class DailyEZ(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEZ.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ez-4' [f64]
    [
        0.211555
        null
        null
        0.978895
        0.682668
        -0.267117
        -0.460222
        -0.969193
    ]
    """

    SUFFIX: ClassVar[str] = "ez"

    def self(self) -> DailyEZ:
        return self

    def encode_type(self, params: DailyEZ.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEZ.Params):
        return params.encode_ewm()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        params = self.params
        mu = pl.col(VALUE).ewm_mean(
            span=params.span,
            alpha=params.alpha,
            min_samples=1,  # TODO: or more?
            ignore_nulls=True,
        )
        sig = pl.col(VALUE).ewm_std(
            span=params.span,
            alpha=params.alpha,
            min_samples=1,  # TODO: or more?
            ignore_nulls=True,
        )
        return ((pl.col(VALUE) - mu) / sig).fill_nan(None)

daily_ez = new(DailyEZ, DailyEZ.Params)

@dataclass(frozen=True)
class DailyEZW(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEZW.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ezw-4' [f64]
    [
        0.144868
        null
        null
        0.981491
        0.843755
        -0.203192
        -0.385844
        -1.153749
    ]
    """

    SUFFIX: ClassVar[str] = "ezw"

    def self(self) -> DailyEZW:
        return self

    def encode_type(self, params: DailyEZW.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEZW.Params):
        return params.encode_ewm()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return self._calc_expr(
            df,
            event,
        )

    def _calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        params = self.params
        mu = pl.col(VALUE).ewm_mean(
            span=params.span,
            alpha=params.alpha,
            min_samples=1,  # TODO: or more?
            ignore_nulls=True,
        )
        sig = (
            pl.col(VALUE)
            .forward_fill()
            .diff()
            .pipe(apis.pl.null_mask, pl.col(VALUE))
            .ewm_std(
                span=params.span,
                alpha=params.alpha,
                min_samples=1,  # TODO: or more?
                ignore_nulls=True,
            )
        )
        return ((pl.col(VALUE) - mu) / sig).fill_nan(None)

daily_ezw = new(DailyEZW, DailyEZW.Params)

# -------------------------------------------------------------


@dataclass(frozen=True)
class DailyERZ(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyERZ.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-erz-4-0.5' [f64]
    [
        1.025375
        null
        null
        1.933986
        1.943604
        -0.685913
        -1.126116
        -1.792186
    ]
    """
    SUFFIX: ClassVar[str] = "erz"

    @dataclass(frozen=True)
    class Params(DailyEM.Params):
        width: float = 0.5
        symmetric: bool = True
        rescale: bool = False

    parent: str | Daily
    params: Params

    def self(self) -> DailyERZ:
        return self

    def encode_type(self, params: DailyERZ.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyERZ.Params):
        tags: list = [params.encode_ewm()]
        if params.width is not None:
            tags.append(params.width)
        if params.symmetric is not None:
            tags.append("sym")
        if params.rescale is not None:
            tags.append("norm")
        return "-".join(map(str, tags))

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params

        assert params.width == 0.5, params

        if params.rescale:
            scale = apis.stats.iqr_ratio(width=params.width)
        else:
            scale = 1

        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()

        mx = calc_max(v, is_null, alpha=params.calc_alpha())
        mn = calc_min(v, is_null, alpha=params.calc_alpha())

        v_ex = pl.col("OLD")

        df = pl.DataFrame(
            {
                "OLD": s,
                "max": mx,
                "min": mn,
            }
        ).with_columns(
            (pl.col("max") - pl.col("min"))
            .mul(1 / 2)
            .mul(scale)
            .alias("rng")
        )

        if params.symmetric:
            mid = (pl.col("max") + pl.col("min")) / 2
        else:
            mid = v_ex.ewm_mean(
                alpha=params.calc_alpha(),
                min_samples=1,  # TODO: or more?
                ignore_nulls=True,
            )

        df = df.with_columns(
            ((v_ex - mid) / pl.col("rng")).alias(VALUE)
        )
        return df.with_columns(
            pl.when(v_ex.is_null())
            .then(None)
            .otherwise(pl.col(VALUE))
            .fill_nan(None)
            .alias(VALUE)
        )

daily_erz = new(DailyERZ, DailyERZ.Params)

@dataclass(frozen=True)
class DailyERZW(DailyERZ):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyERZW.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    """

    @dataclass(frozen=True)
    class Params(DailyERZ.Params):
        pass

    parent: str | Daily
    params: Params

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        raise ValueError((self))

daily_erzw = new(DailyERZW, DailyERZW.Params)

# ----


@dataclass(frozen=True)
class DailyEBeta(DailyEM):

    SUFFIX: ClassVar["str"] = "ebeta"

    @dataclass(frozen=True)
    class Params(DailyEM.Params):
        # span = corr

        rms: str | float = 1.0

        bench: str = ""
        alias: str | None = None
        bench_rms: str | float = 1.0

        rms_scale: float = 1.0
        bench_scale: float = 1.0

    parent: str | Daily
    params: Params

    def self(self) -> DailyEM:
        return self
    
    def get_parent(self) -> Daily:
        return get_parent(self)

    def encode_type(self, params: DailyEBeta.Params):
        return self.SUFFIX
    
    def encode_params(self, params: DailyEBeta.Params):
        tags: list = [params.encode_ewm()]
        if params.alias is not None:
            tags.append(params.alias)
            return "-".join(tags)
        
        tags.append(params.rms)
        
        if params.bench != "":
            tags.append(params.bench)
        
        tags.extend([
            params.bench_rms,
            params.rms_scale,
            params.bench_scale,
        ])
        return "-".join(map(str, tags))

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        bench = daily_pl_df(
            context.get().get(params.bench, Daily),
            alias="bench",
        )
        dfs = [df, bench]
        # if isinstance(params.rms, str):
        #     rms = daily_pl_df(
        #         context.get().get(params.rms, Daily),
        #         domain,
        #         alias="rms"
        #     )
        #     dfs.append(rms)
        # else:
        #     df = df.with_columns(
        #         pl.lit(params.rms)
        #         .alias("rms")
        #     )
        if isinstance(params.bench_rms, str):
            bench_rms = daily_pl_df(
                context.get().get(params.bench_rms, Daily),
                alias="bench_rms",
            )
            dfs.append(bench_rms)
        else:
            df = df.with_columns(
                pl.lit(params.bench_rms).alias("bench_rms")
            )
        df = apis.pl.join_dfs(dfs, align=True, on="date")
        df = df.with_columns(
            # pl.col("rms")
            # .mul(params.rms_scale)
            # .alias("rms"),
            pl.col("bench_rms")
            .mul(params.bench_scale)
            .alias("bench_rms"),
        )
        rx_mu = pl.col(VALUE).ewm_mean(
            span=params.span,
            alpha=params.alpha,
            min_samples=2,  # TODO: or more?
            ignore_nulls=True,
        )
        bench_mu = pl.col("bench").ewm_mean(
            span=params.span,
            alpha=params.alpha,
            min_samples=2,  # TODO: or more?
            ignore_nulls=True,
        )
        cov = (
            (pl.col(VALUE) - rx_mu)
            .mul(pl.col("bench") - bench_mu)
            .ewm_mean(
                span=params.span,
                alpha=params.alpha,
                min_samples=2,  # TODO: or more?
                ignore_nulls=True,
            )
        )
        beta = cov.mul(1 / pl.col("bench_rms").pow(2))
        # corr = cov.mul(
        #     1 / (
        #         pl.col("rms")
        #         .mul(pl.col("bench_rms"))
        #     )
        # )
        # vol_ratio = pl.col("rms") / pl.col("bench_rms")
        # beta = corr.mul(vol_ratio)
        return df.select(
            "date",
            beta.pipe(
                apis.pl.null_mask, pl.col(VALUE)
            ).alias(VALUE),
        )
    
daily_ebeta = new(DailyEBeta, DailyEBeta.Params)

# ----


@dataclass(frozen=True)
class DailyEWmThold(DailyEM):
    
    SUFFIX: ClassVar[str] = "ewmthold"

    @dataclass(frozen=True)
    class Params(DailyEM.Params):
        field: str = "len"

        threshold: float = -2.0
        reset: float = 1.0

        reentry: float | None = 1.0
        rethreshold: float = 1.0

        log: bool = True

    parent: str | Daily
    params: Params

    def self(self) -> DailyEWmThold:
        return self
    
    def get_parent(self) -> Daily:
        return get_parent(self)

    def encode_type(self, params: DailyEWmThold.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEWmThold.Params):
        tags: list = [params.encode_ewm()]
        tags.extend([
            params.field,
            params.threshold,
            params.reset,
        ])
        if params.reentry == True:
            tags.append("re")
        elif params.reentry is not None:
            tags.append(params.reentry)
            tags.append(params.rethreshold)
        if params.log:
            tags.append(params.log)
        return "-".join(map(str, tags))

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        return self._calc_df(
            df,
            event,
        )

    def _calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        params = self.params
        s = df.get_column(VALUE)
        is_null = s.is_null().to_numpy()
        if params.cum:
            # inverse of osc etc.
            s = s.fill_nan(None).forward_fill().diff()
        v = s.fill_null(0).to_numpy()
        res = calc_wm_thold(
            v,
            is_null,
            alpha=params.calc_alpha(),
            threshold=params.threshold,
            reset=params.reset,
            reentry=params.reentry,
            rethreshold=params.rethreshold,
            field=assert_is(
                {
                    "len": 0,
                    "net": 1,
                    "grs": 2,
                }.get(params.field, None)
            ),
            log=params.log,
        )
        return pl.DataFrame(
            {
                "OLD": s,
                VALUE: pl.Series(VALUE, res).fill_nan(None),
            }
        ).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

daily_ewmthold = new(DailyEWmThold, DailyEWmThold.Params)

from .....apis.calcs import calc_metrics_ewm


@nb.jit()
def calc_wm_thold(
    v: np.ndarray,
    is_null: np.ndarray,
    alpha: float,
    threshold: float,
    reset: float,
    reentry: float | None,  # = 2.
    rethreshold: float,
    field: int,
    log: bool,
):
    span = (2 / alpha) - 1
    assert alpha == (2 / (span + 1)), (
        alpha,
        (2 / (span + 1)),
        span,
    )

    n_t = v.shape[0]
    (
        nd_rx_mu,
        nd_rx_cum,
        nd_rx_cmu,
        nd_rx_max,
        nd_rx_min,
        nd_rx_hwm,
        nd_rx_lwm,
        nd_is_hwm,
        nd_is_lwm,
        nd_hwm_len,
        nd_lwm_len,
        nd_hwm_net,
        nd_hwm_grs,
        nd_lwm_net,
        nd_lwm_grs,
        nd_osc_15,
        nd_osc_30,
    ) = calc_metrics_ewm(n_t, v, is_null, span)

    if field == 0:
        nd_hwm = nd_hwm_len
        nd_lwm = nd_lwm_len
    elif field == 1:
        nd_hwm = nd_hwm_net
        nd_lwm = nd_lwm_net
    elif field == 2:
        nd_hwm = nd_hwm_grs
        nd_lwm = nd_lwm_grs
    else:
        raise ValueError(field)

    log_shift = 0
    if log:
        if field == 0:
            log_shift = 0
        else:
            log_shift = 1

    hwm_entry = 0

    res = np.zeros(nd_rx_mu.shape)

    wm_count = 0
    wm_min_cumsum = np.nan

    v_res = np.nan

    re_enter = reentry is not None

    t = 0
    while t < n_t:
        if is_null[t]:
            res[t] = np.nan
            t += 1
            continue

        v_hwm = nd_hwm[t]
        v_lwm = nd_lwm[t]

        if field == 1:
            if v_hwm > 0:
                v_hwm = 0
            if v_lwm < 0:
                v_lwm = 0

        if log:
            if v_hwm < 0:
                v_hwm = -1 * (
                    math.log((v_hwm * -1) + log_shift)
                )
            else:
                v_hwm = math.log(v_hwm + log_shift)
            if v_lwm < 0:
                v_lwm = -1 * (
                    math.log((v_lwm * -1) + log_shift)
                )
            else:
                v_lwm = math.log(v_lwm + log_shift)

        v_hwm_abs = abs(v_hwm)
        v_lwm_abs = abs(v_lwm)

        if v_hwm_abs > v_lwm_abs:
            v_wm_max = v_hwm_abs
            v_wm_min = v_lwm_abs
        else:
            v_wm_max = v_lwm_abs
            v_wm_min = v_hwm_abs

        if wm_count == 0:
            wm_min_cumsum = v_wm_min * v_wm_min
        else:
            wm_min_cumsum += v_wm_min * v_wm_min

        wm_count += 1
        wm_min_rms = math.sqrt(wm_min_cumsum / wm_count)

        v_rms = wm_min_rms

        if v_rms == 0 or wm_count < 3:
            res[t] = np.nan
            t += 1
            continue

        threshold_rms = abs(threshold) * v_rms
        reset_rms = reset * v_rms
        reset_rms_div = v_rms if reset == 0 else reset_rms

        if reentry is not None:
            reentry_rms = reentry * v_rms
        else:
            reentry_rms = 0
            # TODO: assert is not 0 below?

        if threshold == 0:
            v_res = (v_lwm_abs - v_hwm_abs) / v_rms

        elif threshold < 0:
            in_dd = v_hwm_abs > threshold_rms
            in_re_dd = v_hwm_abs > (
                hwm_entry + (v_rms * rethreshold)
            )

            if (
                re_enter
                and in_dd
                and v_lwm_abs > reentry_rms
                and not in_re_dd
            ):
                v_res = (v_lwm_abs - reentry_rms) / v_rms
                hwm_entry = v_hwm_abs
            elif re_enter and in_dd and in_re_dd:
                v_res = -1 * (v_hwm_abs - reset_rms) / v_rms
            elif in_dd:
                v_res = -1 * (v_hwm_abs - reset_rms) / v_rms
            # elif not in_dd:
            #     pass
            elif v_res < 0 and v_hwm_abs < reset_rms:
                v_res = (threshold_rms - v_hwm_abs) / v_rms
                hwm_entry = 0
            elif v_res < 0 and v_hwm_abs >= reset_rms:
                v_res = -1 * (v_hwm_abs - reset_rms) / v_rms
            elif v_res >= 0:
                v_res = (threshold_rms - v_hwm_abs) / v_rms
            elif np.isnan(v_res):
                v_res = (threshold_rms - v_hwm_abs) / v_rms

            else:
                raise ValueError(
                    (
                        t,
                        v_hwm_abs,
                        v_lwm_abs,
                        threshold_rms,
                        reset_rms,
                        in_dd,
                        v_res,
                    )
                )
        else:
            raise ValueError(threshold)

        res[t] = v_res
        t += 1

    return res


# ----


@dataclass(frozen=True)
class DailyEOsc(DailyEM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyEOsc.new(ord, fast=4, slow=12, vol=8)
    >>> unit = DailyEOsc.new(ord, fast=4, slow=12, vol=8, activation="tanh")
    >>> Graph().with_objs(obj, unit).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eosc-4-12-8' [f64]
    [
        -0.418839
        null
        null
        -0.033027
        0.190841
        0.105384
        -0.02915
        -0.430837
    ]
    >>> unit.value_key_val().get(unit.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eosc-4-12-8-tanh' [f64]
    [
        -0.395952
        null
        null
        -0.033015
        0.188558
        0.104996
        -0.029142
        -0.40602
    ]
    """
    SUFFIX: ClassVar[str] = "eosc"

    @dataclass(frozen=True)
    class Params(DailyEM.Params):

        slow: int | None = None
        vol: int | None = None
        activation: str | None = None

        log: bool | None = None

    parent: (
        str
        | Daily
        | FutDailyOrdinal
        | FutDailyRx
        | EqDailyEtf
        | EqDailyRx
        | IndexDailyPx
        | IndexDailyRx
        | FxDailyPx
        | FxDailyRx
    )
    params: Params

    def get_parent(self) -> (
        Daily
        | FutDailyOrdinal
        | FutDailyRx
        | EqDailyEtf
        | EqDailyRx
        | IndexDailyPx
        | IndexDailyRx
        | FxDailyPx
        | FxDailyRx
    ):
        return get_parent(self)

    def self(self) -> DailyEOsc:
        return self
    
    def encode_type(self, params: DailyEOsc.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyEOsc.Params):
        tags: list = [params.encode_ewm()]
        fast = (
            1
            if params.span is None and params.alpha is None
            else params.span
            if params.alpha is None
            else params.alpha
            if params.span is None
            else None
        )
        assert fast is not None, fast
        s_slow = "ex" if params.slow is None else params.slow
        s_vol = "ex" if params.vol is None else params.vol
        
        tags.extend([fast, s_slow, s_vol])

        if params.activation is not None and params.activation != "":
            tags.append(params.activation)
        if params.log:
            tags.append("log")
        return "-".join(map(str, tags))

    def calc_df(
        self,
        df: pl.DataFrame,
        event: Event,
    ) -> pl.DataFrame:
        obj  = self

        params = self.params
        parent = obj.get_parent()
        if isinstance(
            parent,
            (
                FutDailyRx,
                EqDailyRx,
            ),
        ):
            if isinstance(parent, FutDailyRx):
                tags = parent.get_parent().params
                adjust = tags.adjust
            else:
                adjust = "pct"
            rx = pl.col(VALUE)
            if adjust == "pct":
                px = (1 + rx).cum_prod().pipe(null_mask, rx)
            elif adjust == "abs":
                px = rx.cum_sum().pipe(null_mask, rx)
            else:
                raise ValueError(obj)
        elif isinstance(
            parent,
            (
                FutDailyOrdinal,
                EqDailyEtf,
            ),
        ):
            if isinstance(parent, FutDailyOrdinal):
                tags = parent.params
                adjust = tags.adjust
            else:
                adjust = "pct"
            px = pl.col(VALUE)
            if adjust == "pct":
                rx = (
                    px.forward_fill()
                    .pct_change()
                    .pipe(null_mask, px)
                )
            elif adjust == "abs":
                rx = (
                    px.forward_fill()
                    .diff()
                    .pipe(null_mask, px)
                )
            else:
                raise ValueError(obj)
        elif isinstance(parent, Daily):
            if params.cum:
                rx = pl.col(VALUE)
                px = (
                    rx.fill_nan(None)
                    .cum_sum()
                    .pipe(null_mask, rx)
                )
            else:
                px = pl.col(VALUE)
                rx = (
                    px.fill_nan(None)
                    .forward_fill()
                    .diff()
                    .pipe(null_mask, px)
                )
        else:
            raise ValueError((obj, parent))

        df = df.select(VALUE=pl.col(VALUE), px=px, rx=rx)
        px = pl.col("px")
        rx = pl.col("rx")

        if params.slow is None:
            px = rx

        px_fast = assert_is(
            px
            if (
                params.span is None and params.alpha is None
            )
            or params.span == 1
            else px.ewm_mean(
                span=params.span,
                min_samples=2,
                ignore_nulls=True,
            )
            if params.span is not None
            else px.ewm_mean(
                alpha=params.alpha,
                min_samples=2,
                ignore_nulls=True,
            )
            if params.alpha is not None
            else None
        )
        px_slow = (
            None
            if params.slow is None
            else px.rolling_mean(
                len(df),
                min_samples=2,
                # ignore_nulls=True,
            )
            if params.slow == 0
            else px.ewm_mean(
                span=params.slow,
                min_samples=2,
                ignore_nulls=True,
            )
        )
        if params.log:
            px_fast = px_fast.pipe(apis.pl.signed_log)
            if px_slow is not None:
                px_slow = px_slow.pipe(apis.pl.signed_log)

        # TODO: or default to rms?

        rx_vol = (
            rx.rolling_std(
                len(df),
                min_samples=2,
                # ignore_nulls=True,
            )
            if params.vol is None
            else px.ewm_std(
                span=params.vol,
                min_samples=2,
                ignore_nulls=True,
            )
        )

        e = (
            (
                (
                    px_fast
                    - (0 if px_slow is None else px_slow)
                )
                / rx_vol
            )
            .fill_nan(None)
            .pipe(null_mask, pl.col(VALUE))
        )

        # TODO: norm parametrisation (eg. window, centre, etc.)
        df = df.select(e.alias(VALUE))

        e = pl.col(VALUE)

        if px_slow is not None:
            e_norm = (
                e.pow(2)
                .rolling_mean(
                    window_size=len(df), min_samples=10
                )
                .sqrt()
                .fill_nan(None)
            )
            e = e / e_norm

        if (
            params.activation is None
            or params.activation == ""
        ):
            pass
        elif params.activation == "tanh":
            e = e.tanh()
        else:
            raise ValueError(obj)

        df = df.select(
            e.fill_nan(None)
            .pipe(null_mask, pl.col(VALUE))
            .alias(VALUE)
        )
        # log("INFO")( "end osc")
        return df

daily_eosc = new(DailyEOsc, DailyEOsc.Params)

# TODO: encode ..., self, new, all

#  ------------------

__all__ = [
    "P",
    "O",
    "DailyEM",
    "DailyEMA",
    "daily_ema",
    "DailyEMax",
    "daily_emax",
    "calc_max",
    "DailyEMid",
    "daily_emid",
    "DailyEMin",
    "daily_emin",
    "calc_min",
    "DailyEPLast",
    "daily_eplast",
    "calc_p_last",
    "DailyERms",
    "daily_erms",
    "DailyERng",
    "daily_erng",
    "DailyESDev",
    "daily_esdev",
    "DailyEVar",
    "daily_evar",
    "DailyEZ",
    "daily_ez",
    "DailyERZ",
    "daily_erz",
    "DailyEZW",
    "daily_ezw",
    "DailyERZW",
    "daily_erzw",
    "DailyEHwm",
    "daily_ehwm",
    "calc_hwm",
    "daily_ehwm",
    "DailyELwm",
    "calc_lwm",
    "DailyEWmThold",
    "calc_wm_thold",
    "daily_ewmthold",
    "DailyEBeta",
    "daily_ebeta",
    "DailyEOsc",
    "daily_eosc",
]
