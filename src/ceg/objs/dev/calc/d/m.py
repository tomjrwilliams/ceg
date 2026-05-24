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

from ..... import apis
from .....apis import t, kv, assert_is
from .....apis.pl import *

from .....engine import *

from ...bars.d import *

P = ParamSpec("P")
O = TypeVar("O", bound=Object)

#  ------------------


@dataclass(frozen=True)
class DailyM(
    Daily,
    DependsParent,
    HasDataFrameValue
):

    SUFFIX: ClassVar[str] = "M"
    # NOTE: to override

    @dataclass(frozen=True)
    class Params(Daily.Params):
        window: int | None

        ann: bool = False

        cum: bool = False

        def encode_m(self):
            tags: list
            if self.window is None:
                tags = ["ex"]
            else:
                tags = [self.window]
            if self.ann:
                tags.append("ann")
            if self.cum:
                tags.append("cum")
            return "-".join(map(str, tags))

    parent: str | Daily
    params: Params

    def get_parent(self) -> Daily:
        return get_parent(self)

    def self(self) -> DailyM:
        return self
    
    # def encode_type(self, params: DailyM.Params):
    #     return self.SUFFIX

    # def encode_params(self, params: DailyM.Params):
    #     return params.encode_m()

    def calc_expr(self, df: pl.DataFrame, event: ValueUpdate):
        raise NotImplementedError(self)
    
    def calc_df(self, df: pl.DataFrame, event: ValueUpdate):
        raise NotImplementedError(self)

    def handle(self, event: Event, snap: Snap):
        assert self.SUFFIX != "M", self
        
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
class DailyMA(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMA.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ma-4' [f64]
    [
        5707.5
        null
        null
        5766.875
        5821.0
        5800.5
        5785.1875
        5737.0625
    ]
    """
    SUFFIX: ClassVar[str] = "ma"

    def self(self) -> DailyMA:
        return self
    
    def encode_type(self, params: DailyMA.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyMA.Params):
        return params.encode_m()

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
        return pl.col(VALUE).rolling_mean(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )

daily_ma = new(DailyMA, DailyMA.Params)

@dataclass(frozen=True)
class DailyKurt(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyKurt.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-kurt-4' [f64]
    [
        -1.116847
        null
        null
        -2.0
        -2.0
        -1.5
        -1.070286
        -1.379246
    ]
    """

    SUFFIX: ClassVar[str] = "kurt"


    def self(self) -> DailyKurt:
        return self
    
    def encode_type(self, params: DailyKurt.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyKurt.Params):
        return params.encode_m()

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
        return pl.col(VALUE).rolling_kurtosis(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )

daily_kurt = new(DailyKurt, DailyKurt.Params)

@dataclass(frozen=True)
class DailyMax(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMax.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-max-4' [f64]
    [
        5729.75
        null
        null
        5815.5
        5826.5
        5826.5
        5826.5
        5826.5
    ]
    """

    SUFFIX: ClassVar[str] = "max"

    def self(self) -> DailyMax:
        return self
    
    def encode_type(self, params: DailyMax.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyMax.Params):
        return params.encode_m()

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
        return pl.col(VALUE).rolling_max(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )

daily_max = new(DailyMax, DailyMax.Params)

@dataclass(frozen=True)
class DailyMed(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMed.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-med-4' [f64]
    [
        5715.5
        null
        null
        5766.875
        5821.0
        5815.5
        5787.5
        5749.375
    ]
    """


    SUFFIX: ClassVar[str] = "med"

    def self(self) -> DailyMed:
        return self
    
    def encode_type(self, params: DailyMed.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyMed.Params):
        return params.encode_m()

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
        return pl.col(VALUE).rolling_median(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )

daily_med = new(DailyMed, DailyMed.Params)

@dataclass(frozen=True)
class DailyMin(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMin.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-min-4' [f64]
    [
        5669.25
        null
        null
        5718.25
        5815.5
        5759.5
        5739.25
        5623.0
    ]
    """
    SUFFIX: ClassVar[str] = "min"

    def self(self) -> DailyMin:
        return self
    
    def encode_type(self, params: DailyMin.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyMin.Params):
        return params.encode_m()

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

        return pl.col(VALUE).rolling_min(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
    
daily_min = new(DailyMin, DailyMin.Params)


@dataclass(frozen=True)
class DailyPLast(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyPLast.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-plast-4' [f64]
    [
        0.333333
        null
        null
        0.0
        0.0
        0.666667
        1.0
        1.0
    ]
    """

    SUFFIX: ClassVar[str] = "plast"

    def self(self) -> DailyPLast:
        return self
    
    def encode_type(self, params: DailyPLast.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyPLast.Params):
        return params.encode_m()
    
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
            l=len(df)
            if params.window is None
            else params.window,
            norm=False,  # TODO: parametrise in path?
        )

        return pl.DataFrame({"mask": s, VALUE: res}).select(
            pl.when(pl.col("mask").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

daily_plast = new(DailyPLast, DailyPLast.Params)

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
                        else 0.5
                        if v[ij] == loc[i]
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
class DailyRank(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRank.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rank-4' [f64]
    [
        3.0
        null
        null
        2.0
        2.0
        1.0
        1.0
        1.0
    ]
    """

    SUFFIX: ClassVar[str] = "rank"

    def self(self) -> DailyRank:
        return self
    
    def encode_type(self, params: DailyRank.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyRank.Params):
        return params.encode_m()

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
        return pl.col(VALUE).rolling_rank(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
    
daily_rank = new(DailyRank, DailyRank.Params)


@dataclass(frozen=True)
class DailyRms(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRms.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-rms-4' [f64]
    [
        0.007817
        null
        null
        0.012045
        0.0121
        0.011903
        0.010457
        0.011816
    ]
    """

    SUFFIX: ClassVar[str] = "rms"

    def self(self) -> DailyRms:
        return self
    
    def encode_type(self, params: DailyRms.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyRms.Params):
        return params.encode_m()

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
        return (
            pl.col(VALUE)
            .pow(2)
            .rolling_mean(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            .sqrt()
        )

daily_rms = new(DailyRms, DailyRms.Params)

@dataclass(frozen=True)
class DailySDev(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailySDev.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-sdev-4' [f64]
    [
        0.009001
        null
        null
        0.011345
        0.010688
        0.014262
        0.012023
        0.009659
    ]
    """

    SUFFIX: ClassVar[str] = "sdev"

    def self(self) -> DailySDev:
        return self
    
    def encode_type(self, params: DailySDev.Params):
        return self.SUFFIX

    def encode_params(self, params: DailySDev.Params):
        return params.encode_m()

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
        return pl.col(VALUE).rolling_std(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
    
daily_sdev = new(DailySDev, DailySDev.Params)


@dataclass(frozen=True)
class DailySkew(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailySkew.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-skew-4' [f64]
    [
        0.15138
        null
        null
        2.6070e-14
        3.3046e-14
        0.073938
        0.464041
        -0.228863
    ]
    """
    SUFFIX: ClassVar[str] = "skew"

    def self(self) -> DailySkew:
        return self
    
    def encode_type(self, params: DailySkew.Params):
        return self.SUFFIX

    def encode_params(self, params: DailySkew.Params):
        return params.encode_m()
    
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
        return pl.col(VALUE).rolling_skew(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )

daily_skew = new(DailySkew, DailySkew.Params)

@dataclass(frozen=True)
class DailyVar(DailyM):
    """
    >>> ord = FutDailyRx.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyVar.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-var-4' [f64]
    [
        0.000081
        null
        null
        0.000129
        0.000114
        0.000203
        0.000145
        0.000093
    ]
    """
    SUFFIX: ClassVar[str] = "var"

    def self(self) -> DailyVar:
        return self
    
    def encode_type(self, params: DailyVar.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyVar.Params):
        return params.encode_m()
    
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
        return pl.col(VALUE).rolling_var(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )

daily_var = new(DailyVar, DailyVar.Params)

@dataclass(frozen=True)
class DailyZ(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyZ.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-z-4' [f64]
    [
        0.406191
        null
        null
        0.707107
        0.707107
        -1.141092
        -1.083191
        -1.346812
    ]
    """


    SUFFIX: ClassVar[str] = "z"

    def self(self) -> DailyZ:
        return self
    
    def encode_type(self, params: DailyZ.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyZ.Params):
        return params.encode_m()

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
        mu = pl.col(VALUE).rolling_mean(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        sig = pl.col(VALUE).rolling_std(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        return ((pl.col(VALUE) - mu) / sig).fill_nan(None)

daily_z = new(DailyZ, DailyZ.Params)

@dataclass(frozen=True)
class DailyZW(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyZW.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-zw-4' [f64]
    [
        0.406191
        null
        null
        0.707107
        0.707107
        -1.141092
        -1.083191
        -1.346812
    ]
    """

    SUFFIX: ClassVar[str] = "zw"

    def self(self) -> DailyZW:
        return self
    
    def encode_type(self, params: DailyZW.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyZW.Params):
        return params.encode_m()
    
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
        mu = pl.col(VALUE).rolling_mean(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        sig = (
            pl.col(VALUE)
            .forward_fill()
            .diff()
            .pipe(apis.pl.null_mask, pl.col(VALUE))
            .rolling_std(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        return ((pl.col(VALUE) - mu) / sig).fill_nan(None)

daily_zw = new(DailyZW, DailyZW.Params)

# ----------------------------------

@dataclass(frozen=True)
class DailyMid(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyMid.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-mid-4' [f64]
    [
        5699.5
        null
        null
        5766.875
        5821.0
        5793.0
        5782.875
        5724.75
    ]
    """

    SUFFIX: ClassVar[str] = "mid"

    @dataclass(frozen=True)
    class Params(DailyM.Params):
        width: float | None = None

    parent: str | Daily
    params: Params

    def self(self) -> DailyMid:
        return self
    
    def encode_type(self, params: DailyMid.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyMid.Params):
        tags = [params.encode_m()]
        if params.width is None:
            return tags[0]
        return "-".join(map(str, tags + [params.width]))
    
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
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
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
    
daily_mid = new(DailyMid, DailyMid.Params)

@dataclass(frozen=True)
class DailyRng(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRng.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rng-4' [f64]
    [
        60.5
        null
        null
        97.25
        11.0
        67.0
        87.25
        203.5
    ]
    """

    SUFFIX: ClassVar[str] = "rng"

    @dataclass(frozen=True)
    class Params(DailyM.Params):
        width: float | None = None

    parent: str | Daily
    params: Params

    def self(self) -> DailyRng:
        return self
    
    def encode_type(self, params: DailyRng.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyRng.Params):
        tags = [params.encode_m()]
        if params.width is None:
            return tags[0]
        return "-".join(map(str, tags + [params.width]))
    
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
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
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


daily_rng = new(DailyRng, DailyRng.Params)

@dataclass(frozen=True)
class DailyQile(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyQile.new(ord, window=4, quantile=.5)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-qile-4-0.5' [f64]
    [
        5718.25
        null
        null
        5815.5
        5826.5
        5815.5
        5815.5
        5759.5
    ]
    """


    SUFFIX: ClassVar[str] = "qile"

    @dataclass(frozen=True)
    class Params(DailyM.Params):
        quantile: float | None = None

    parent: str | Daily
    params: Params

    def self(self) -> DailyQile:
        return self
    
    def encode_type(self, params: DailyQile.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyQile.Params):
        tags = [params.encode_m()]
        if params.quantile is None:
            raise ValueError(params)
        return "-".join(map(str, tags + [params.quantile]))

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
        assert params.quantile is not None, self

        return pl.col(VALUE).rolling_quantile(
            params.quantile,
            window_size=len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )

daily_qile = new(DailyQile, DailyQile.Params)


@dataclass(frozen=True)
class DailyRZ(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRZ.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rz-4-0.25-sym' [f64]
    [
        0.285714
        null
        null
        2.0
        2.0
        -3.343284
        -1.512243
        -3.825922
    ]
    """

    SUFFIX: ClassVar[str] = "rz"

    @dataclass(frozen=True)
    class Params(DailyM.Params):
        
        width: float = 0.25
        symmetric: bool = True
        rescale: bool = False

    parent: str | Daily
    params: Params

    def self(self) -> DailyRZ:
        return self
    
    def encode_type(self, params: DailyRZ.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyRZ.Params):
        tags: list = [params.encode_m()]
        if params.width != 0.25:
            tags.append(params.width)
        if params.symmetric:
            tags.append("sym")
        if params.rescale:
            tags.append("norm")
        return "-".join(map(str, tags))
    
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
        assert params.width is not None, params
        v = pl.col(VALUE)
        med = v.rolling_median(
            len(df)
            if params.window is None
            else params.window,
            min_samples=1,  # TODO: or more?
        )
        if params.symmetric:
            q_up = v.rolling_quantile(
                0.5 + params.width,
                interpolation="linear",
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            q_dn = v.rolling_quantile(
                0.5 - params.width,
                interpolation="linear",
                window_size=len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            iqr = q_up - q_dn
            mad = iqr / 2
            # for symmetric, where median=average(up, dn):
            # iqr/2 == mad
        else:
            med = v.rolling_median(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )
            mad = ((v - med).abs()).rolling_median(
                len(df)
                if params.window is None
                else params.window,
                min_samples=1,  # TODO: or more?
            )

        if params.rescale:
            if not params.symmetric:
                mad = mad * 1.4826
            else:
                mad = mad * apis.stats.iqr_ratio(
                    params.width
                )

        return ((v - med) / mad).fill_nan(None)

daily_rz = new(DailyRZ, DailyRZ.Params)

@dataclass(frozen=True)
class DailyRZW(DailyRZ):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyRZW.new(ord, window=4)
    >>> Graph().with_obj(obj)
    """

    SUFFIX: ClassVar[str] = "rzw"

    def self(self) -> DailyRZW:
        return self
    
    def encode_type(self, params: DailyRZW.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyRZW.Params):
        raise ValueError(self)
    
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
        raise NotImplementedError()

daily_rzw = new(DailyRZW, DailyRZW.Params)

@dataclass(frozen=True)
class DailyOsc(DailyM):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyOsc.new(ord, fast=4, slow=12, vol=8)
    >>> unit = DailyOsc.new(ord, fast=4, slow=12, vol=8, activation="tanh")
    >>> Graph().with_objs(obj, unit).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-osc-4-12-8' [f64]
    [
        0.979022
        null
        null
        1.240631
        1.349691
        1.105384
        0.828809
        0.060248
    ]
    >>> unit.value_key_val().get(unit.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-osc-4-12-8-tanh' [f64]
    [
        0.752642
        null
        null
        0.845635
        0.87398
        0.802425
        0.679836
        0.060175
    ]
    """

    @dataclass(frozen=True)
    class Params(DailyM.Params):
        slow: int | None = None
        vol: int | None = None
        activation: str | None = None

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

    def self(self) -> DailyOsc:
        return self

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
    
    def encode_type(self, params: DailyOsc.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyOsc.Params):
        tags: list = [params.encode_m()]
        slow = "ex" if params.slow is None else params.slow
        vol = "ex" if params.vol is None else params.vol
        tags.extend([slow, vol])
        if params.activation is not None and params.activation != "":
            tags.append(params.activation)
        return "-".join(map(str, tags))

    def calc_df(
        self,
        df: pl.DataFrame,
        event: Event,
    ) -> pl.DataFrame:
        obj = self
        params = self.params
        parent = obj.get_parent()
        if isinstance(parent, FutDailyRx):
            tags = parent.get_parent().params
            rx = pl.col(VALUE)
            if tags.adjust == "pct":
                px = (1 + rx).cum_prod().pipe(null_mask, rx)
            elif tags.adjust == "abs":
                px = rx.cum_sum().pipe(null_mask, rx)
            else:
                raise ValueError(obj)
        elif isinstance(parent, FutDailyOrdinal):
            tags = parent.params
            px = pl.col(VALUE)
            if tags.adjust == "pct":
                rx = (
                    px.forward_fill()
                    .pct_change()
                    .pipe(null_mask, px)
                )
            elif tags.adjust == "abs":
                rx = (
                    px.forward_fill()
                    .diff()
                    .pipe(null_mask, px)
                )
            else:
                raise ValueError(obj)
        else:
            raise ValueError((obj, parent))

        df = df.select(VALUE=pl.col(VALUE), px=px, rx=rx)
        px = pl.col("px")
        rx = pl.col("rx")

        px_fast = (
            px
            if params.window is None or params.window == 1
            else px.rolling_mean(
                params.window, min_samples=2
            )
        )
        px_slow = (
            px.rolling_mean(len(df), min_samples=2)
            if params.slow is None
            else px.rolling_mean(params.slow, min_samples=2)
        )
        # TODO: or default to rms?
        rx_vol = (
            rx.rolling_std(len(df), min_samples=2)
            if params.vol is None
            else px.rolling_std(params.vol, min_samples=2)
        )

        e = ((px_fast - px_slow) / rx_vol).pipe(
            null_mask, pl.col(VALUE)
        )
        if params.activation is None:
            pass
        elif params.activation == "tanh":
            e = e.tanh()
        else:
            raise ValueError(obj)

        df = df.select(e.alias(VALUE))
        return df

daily_osc = new(DailyOsc, DailyOsc.Params)

#  ------------------

__all__ = [
    "P",
    "O",
    "DailyM",
    "DailyMA",
    "daily_ma",
    "DailyKurt",
    "daily_kurt",
    "DailyMax",
    "daily_max",
    "DailyMed",
    "daily_med",
    "DailyMid",
    "daily_mid",
    "DailyMin",
    "daily_min",
    "DailyPLast",
    "daily_plast",
    "calc_p_last",
    "daily_qile",
    "DailyQile",
    "DailyRank",
    "daily_rank",
    "DailyRms",
    "daily_rms",
    "DailyRng",
    "DailySDev",
    "daily_sdev",
    "DailySkew",
    "daily_skew",
    "DailyVar",
    "daily_var",
    "DailyZ",
    "daily_z",
    "DailyRZ",
    "daily_rz",
    "DailyZW",
    "daily_zw",
    "DailyRZW",
    "daily_rzw",
    "DailyOsc",
    "daily_osc",
    # TODO: beta, corr, etc.
]
