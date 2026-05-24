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

from .m import *


@dataclass(frozen=True)
class DailyDM(DailyM):
    pass


@dataclass(frozen=True)
class DailyDMA(DailyDM, DailyMA):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDMA.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dma-4' [f64]
    [
        10.75
        null
        null
        48.625
        5.5
        -41.0
        -45.9375
        -114.0625
    ]
    """

    SUFFIX: ClassVar[str] = "dma"

    def self(self) -> DailyDMA:
        return self
    
    def encode_type(self, params: DailyDMA.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDMA.Params):
        return params.encode_m()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dma = new(DailyDMA, DailyDMA.Params)

@dataclass(frozen=True)
class DailyDKurt(DailyDM, DailyKurt):

    SUFFIX: ClassVar[str] = "dkurt"

    def self(self) -> DailyDKurt:
        return self
    
    def encode_type(self, params: DailyDKurt.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDKurt.Params):
        return params.encode_m()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dkurt = new(DailyDKurt, DailyDKurt.Params)

@dataclass(frozen=True)
class DailyDMax(DailyDM, DailyMax):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDMax.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmax-4' [f64]
    [
        -11.5
        null
        null
        0.0
        0.0
        -67.0
        -87.25
        -203.5
    ]
    """

    SUFFIX: ClassVar[str] = "dmax"

    def self(self) -> DailyDMax:
        return self
    
    def encode_type(self, params: DailyDMax.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDMax.Params):
        return params.encode_m()


    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dmax = new(DailyDMax, DailyDMax.Params)

@dataclass(frozen=True)
class DailyDMed(DailyDM, DailyMed):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDMed.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmed-4' [f64]
    [
        2.75
        null
        null
        48.625
        5.5
        -56.0
        -48.25
        -126.375
    ]
    """

    version: int = 0
    type: Literal["DailyDMed"] = "DailyDMed"

    def self(self) -> DailyDMed:
        return self
    
    def encode_type(self, params: DailyDMed.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDMed.Params):
        return params.encode_m()

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dmed = new(DailyDMed, DailyDMed.Params)

@dataclass(frozen=True)
class DailyDMid(DailyDM, DailyMid):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDMid.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmid-4' [f64]
    [
        18.75
        null
        null
        48.625
        5.5
        -33.5
        -43.625
        -101.75
    ]
    """
    SUFFIX: ClassVar[str] = "dmid"

    # TODO: encode params

    def self(self) -> DailyDMid:
        return self
    
    def encode_type(self, params: DailyDMid.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDMid.Params):
        return params.encode_m()

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

daily_dmid = new(DailyDMid, DailyDMid.Params)

@dataclass(frozen=True)
class DailyDMin(DailyDM, DailyMin):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDMin.new(ord, window=4)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmin-4' [f64]
    [
        49.0
        null
        null
        97.25
        11.0
        0.0
        0.0
        0.0
    ]
    """

    SUFFIX: ClassVar[str] = "dmin"

    def self(self) -> DailyDMin:
        return self
    
    def encode_type(self, params: DailyDMin.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDMin.Params):
        return params.encode_m()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dmin = new(DailyDMin, DailyDMin.Params)

@dataclass(frozen=True)
class DailyDPLast(DailyDM, DailyPLast):

    SUFFIX: ClassVar[str] = "dplast"

    def self(self) -> DailyDPLast:
        return self
    
    def encode_type(self, params: DailyDPLast.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDPLast.Params):
        return params.encode_m()
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

daily_dplast = new(DailyDPLast, DailyDPLast.Params)

@dataclass(frozen=True)
class DailyDQile(DailyDM, DailyQile):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = DailyDQile.new(ord, window=4, quantile=.5)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dqile-4-0.5' [f64]
    [
        0.0
        null
        null
        0.0
        0.0
        -56.0
        -76.25
        -136.5
    ]
    """

    SUFFIX: ClassVar[str] = "dqile"


    def self(self) -> DailyQile:
        return self
    
    def encode_type(self, params: DailyQile.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyQile.Params):
        return params.encode_m()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )
    
daily_dqile = new(DailyDQile, DailyDQile.Params)

@dataclass(frozen=True)
class DailyDRank(DailyDM, DailyRank):

    SUFFIX: ClassVar[str] = "drank"

    def self(self) -> DailyDRank:
        return self
    
    def encode_type(self, params: DailyDRank.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDRank.Params):
        return params.encode_m()
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_drank = new(DailyDRank, DailyDRank.Params)

@dataclass(frozen=True)
class DailyDRms(DailyDM, DailyRms):

    SUFFIX: ClassVar[str] = "drms"

    def self(self) -> DailyDRms:
        return self
    
    def encode_type(self, params: DailyDRms.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDRms.Params):
        return params.encode_m()
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_drms = new(DailyDRms, DailyDRms.Params)

@dataclass(frozen=True)
class DailyDRng(DailyDM, DailyRng):

    SUFFIX: ClassVar[str] = "drng"

    def self(self) -> DailyDRng:
        return self
    
    def encode_type(self, params: DailyDRng.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDRng.Params):
        return params.encode_m()
    
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

daily_drng = new(DailyDRng, DailyDRng.Params)

@dataclass(frozen=True)
class DailyDSDev(DailyDM, DailySDev):

    SUFFIX: ClassVar[str] = "dsdev"

    def self(self) -> DailyDSDev:
        return self
    
    def encode_type(self, params: DailyDSDev.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDSDev.Params):
        return params.encode_m()
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dsdev = new(DailyDSDev, DailyDSDev.Params)

@dataclass(frozen=True)
class DailyDSkew(DailyDM, DailySkew):

    SUFFIX: ClassVar[str] = "dskew"

    def self(self) -> DailyDSkew:
        return self
    
    def encode_type(self, params: DailyDSkew.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDSkew.Params):
        return params.encode_m()
    
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )

daily_dskew = new(DailyDSkew, DailyDSkew.Params)

@dataclass(frozen=True)
class DailyDVar(DailyDM, DailyVar):

    SUFFIX: ClassVar[str] = "dvar"

    def self(self) -> DailyDVar:
        return self
    
    def encode_type(self, params: DailyDVar.Params):
        return self.SUFFIX

    def encode_params(self, params: DailyDVar.Params):
        return params.encode_m()
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ValueUpdate,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
        )


daily_dvar = new(DailyDVar, DailyDVar.Params)

#  ------------------

__all__ = [
    # "DailyDMParams",
    "DailyDM",
    "DailyDMA",
    "daily_dma",
    "DailyDMax",
    "daily_dmax",
    "DailyDMed",
    "daily_dmed",
    "DailyDMid",
    "daily_dmid",
    "DailyDMin",
    "daily_dmin",
    "DailyDQile",
    "daily_dqile",
    "daily_drank",
    "DailyDRank",
    "daily_dskew",
    "DailyDSkew",
    "daily_drms",
    "DailyDRms",
    "daily_drng",
    "DailyDRng",
    "daily_dsdev",
    "DailyDSDev",
    "daily_dvar",
    "DailyDVar",
    "daily_dplast",
    "DailyDPLast",
    "daily_dkurt",
    "DailyDKurt",
]
