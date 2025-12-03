
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

@dataclass(frozen=True)
class DailyDEM(DailyEM):
    pass

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyDEM(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class DailyDEMA(DailyDEM):
    version: int = 0
    type: Literal["DailyDEMA"] = "DailyDEMA"

    SUFFIX: ClassVar[str] = "dema"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyDEMASnap"] = "DailyDEMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDEMA:
        return HandleNewDailyDEMA()

@dataclass(frozen=True)
class HandleNewDailyDEMA(
    HandleNewDailyDEM,
    HandleNewDailyEMA
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

@dataclass(frozen=True)
class DailyDEMax(DailyDEM):
    version: int = 0
    type: Literal["DailyDEMax"] = "DailyDEMax"

    SUFFIX: ClassVar[str] = "demax"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyDEMaxSnap"] = "DailyDEMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDEMax:
        return HandleNewDailyDEMax()
@dataclass(frozen=True)
class HandleNewDailyDEMax(
    HandleNewDailyDEM,
    HandleNewDailyEMax
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
            domain,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

@dataclass(frozen=True)
class DailyDEMid(DailyDEM):
    version: int = 0
    type: Literal["DailyDEMid"] = "DailyDEMid"

    SUFFIX: ClassVar[str] = "demid"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyDEMidSnap"] = "DailyDEMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDEMid:
        return HandleNewDailyDEMid()
@dataclass(frozen=True)
class HandleNewDailyDEMid(
    HandleNewDailyDEM,
    HandleNewDailyEMid
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
            domain,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
    
@dataclass(frozen=True)
class DailyDEMin(DailyDEM):
    version: int = 0
    type: Literal["DailyDEMin"] = "DailyDEMin"

    SUFFIX: ClassVar[str] = "demin"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyDEMinSnap"] = "DailyDEMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDEMin:
        return HandleNewDailyDEMin()

@dataclass(frozen=True)
class HandleNewDailyDEMin(
    HandleNewDailyDEM,
    HandleNewDailyEMin
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
            domain,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )


@dataclass(frozen=True)
class HandleNewDailyDEPLast(
    HandleNewDailyDEM,
    HandleNewDailyEPLast
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
            domain,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
    
@dataclass(frozen=True)
class HandleNewDailyDERms(
    HandleNewDailyDEM,
    HandleNewDailyERms
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )


@dataclass(frozen=True)
class HandleNewDailyDERng(
    HandleNewDailyDEM,
    HandleNewDailyERng
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
            domain,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

@dataclass(frozen=True)
class HandleNewDailyDESDev(
    HandleNewDailyDEM,
    HandleNewDailyESDev
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

@dataclass(frozen=True)
class HandleNewDailyDEVar(
    HandleNewDailyDEM,
    HandleNewDailyEVar
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )
    
__all__ = [
    # "DailyDEMParams",
    "DailyDEM",
    "HandleNewDailyDEM",
    "HandleNewDailyDEMA",
    "DailyDEMA",
    "HandleNewDailyDEMax",
    "DailyDEMax",
    "HandleNewDailyDEMid",
    "DailyDEMid",
    "HandleNewDailyDEMin",
    "DailyDEMin",
    "HandleNewDailyDEPLast",
    "HandleNewDailyDERms",
    "HandleNewDailyDERng",
    "HandleNewDailyDESDev",
    "HandleNewDailyDEVar",
]