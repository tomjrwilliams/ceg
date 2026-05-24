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
from .....apis import t
from .....apis import kv
from .....engine import *

from .em import *

#  ------------------


@dataclass(frozen=True)
class DailyDEM(DailyEM):
    pass


#  ------------------


@dataclass(frozen=True)
class DailyDEMA(DailyDEM, DailyEMA):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDEMA.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dema-4' [f64]
    [
        6.699801
        null
        null
        62.36988
        44.021928
        -13.786843
        -20.422106
        -82.003264
    ]
    """


    SUFFIX: ClassVar[str] = "dema"

    def self(self) -> DailyDEMA:
        return self

    def encode_type(self, params: DailyDEMA.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDEMA.Params):
        return params.encode_ewm()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dema = new(DailyDEMA, DailyDEMA.Params)

@dataclass(frozen=True)
class DailyDEMax(DailyDEM, DailyEMax):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDEMax.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-demax-4' [f64]
    [
        -2.82
        null
        null
        0.0
        0.0
        -40.2
        -36.27
        -91.512
    ]
    """

    SUFFIX: ClassVar[str] = "demax"

    def self(self) -> DailyDEMax:
        return self

    def encode_type(self, params: DailyDEMax.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDEMax.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
        df = self._calc_df(
            df,
            event,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

daily_demax = new(DailyDEMax, DailyDEMax.Params)

@dataclass(frozen=True)
class DailyDEMid(DailyDEM, DailyEMid):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDEMid.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-demid-4' [f64]
    [
        3.714
        null
        null
        32.2494
        22.64964
        -20.1
        -18.135
        -45.756
    ]
    """
    SUFFIX: ClassVar[str] = "demid"

    def self(self) -> DailyDEMid:
        return self

    def encode_type(self, params: DailyDEMid.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDEMid.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
        df = self._calc_df(
            df,
            event,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

daily_demid = new(DailyDEMid, DailyDEMid.Params)

@dataclass(frozen=True)
class DailyDEMin(DailyDEM, DailyEMin):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDEMin.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-demin-4' [f64]
    [
        10.248
        null
        null
        64.4988
        45.29928
        0.0
        0.0
        0.0
    ]
    """

    SUFFIX: ClassVar[str] = "demin"

    def self(self) -> DailyDEMin:
        return self

    def encode_type(self, params: DailyDEMin.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDEMin.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:
        df = df.with_columns(pl.col(VALUE).alias("OLD"))
        df = self._calc_df(
            df,
            event,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

daily_demin = new(DailyDEMin, DailyDEMin.Params)

@dataclass(frozen=True)
class DailyDEPLast(DailyDEM, DailyEPLast):

    SUFFIX: ClassVar[str] = "deplast"

    def self(self) -> DailyDEPLast:
        return self

    def encode_type(self, params: DailyDEPLast.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDEPLast.Params):
        return params.encode_ewm()
    

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
        df = self._calc_df(
            df,
            event,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

daily_deplast = new(DailyDEPLast, DailyDEPLast.Params)

@dataclass(frozen=True)
class DailyDERms(DailyDEM, DailyERms):
    SUFFIX: ClassVar[str] = "derms"

    def self(self) -> DailyDERms:
        return self

    def encode_type(self, params: DailyDERms.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDERms.Params):
        return params.encode_ewm()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_derms = new(DailyDERms, DailyDERms.Params)

@dataclass(frozen=True)
class DailyDERng(DailyDEM, DailyERng):

    SUFFIX: ClassVar[str] = "derng"


    def self(self) -> DailyDERng:
        return self

    def encode_type(self, params: DailyDERng.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDERng.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
        df = self._calc_df(
            df,
            event,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

daily_derng = new(DailyDERng, DailyDERng.Params)

@dataclass(frozen=True)
class DailyDEHwm(DailyDEM, DailyEHwm):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDEHwm.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dehwm-4' [f64]
    [
        -2.82
        null
        null
        0.0
        0.0
        -40.2
        -36.27
        -91.512
    ]
    """

    SUFFIX: ClassVar[str] = "dehwm"


    def self(self) -> DailyDEHwm:
        return self

    def encode_type(self, params: DailyDEHwm.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDEHwm.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
        df = self._calc_df(
            df,
            event,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

daily_dehwm = new(DailyDEHwm, DailyDEHwm.Params)

@dataclass(frozen=True)
class DailyDELwm(DailyDEM, DailyELwm):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDELwm.new(ord, span=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-delwm-4' [f64]
    [
        -2.82
        null
        null
        0.0
        0.0
        -40.2
        -36.27
        -91.512
    ]
    """

    SUFFIX: ClassVar[str] = "delwm"

    def self(self) -> DailyDELwm:
        return self

    def encode_type(self, params: DailyDELwm.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDELwm.Params):
        return params.encode_ewm()
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
        df = self._calc_df(
            df,
            event,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

daily_delwm = new(DailyDELwm, DailyDELwm.Params)

@dataclass(frozen=True)
class DailyDESDev(DailyDEM, DailyESDev):

    SUFFIX: ClassVar[str] = "desdev"


    def self(self) -> DailyDESDev:
        return self

    def encode_type(self, params: DailyDESDev.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDESDev.Params):
        return params.encode_ewm()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_desdev = new(DailyDESDev, DailyDESDev.Params)

@dataclass(frozen=True)
class DailyDEVar(DailyDEM, DailyEVar):
    SUFFIX: ClassVar[str] = "devar"

    def self(self) -> DailyDEVar:
        return self

    def encode_type(self, params: DailyDEVar.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDEVar.Params):
        return params.encode_ewm()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_devar = new(DailyDEVar, DailyDEVar.Params)

__all__ = [
    # "DailyDEMParams",
    "DailyDEM",
    "DailyDEMA",
    "daily_dema",
    "DailyDEMax",
    "daily_demax",
    "DailyDEMid",
    "daily_demid",
    "DailyDEMin",
    "daily_demin",
    "DailyDEPLast",
    "daily_dplast",
    "DailyDERms",
    "daily_erms",
    "DailyDERng",
    "daily_erng",
    "DailyDEHwm",
    "daily_dehwm",
    "DailyDELwm",
    "daily_delwm",
    "DailyDESDev",
    "daily_desdev",
    "DailyDEVar",
    "daily_devar",
]
