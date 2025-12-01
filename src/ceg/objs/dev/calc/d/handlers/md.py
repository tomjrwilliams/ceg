
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

from ...... import apis
from ......apis import t
from ......apis import kv
from ......engine import *

from ..m import *
from .m import *

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyDM(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyDMA(
    HandleNewDailyDM,
    HandleNewDailyMA,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )
    
@dataclass(frozen=True)
class HandleNewDailyDKurt(
    HandleNewDailyDM,
    HandleNewDailyKurt,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )


@dataclass(frozen=True)
class HandleNewDailyDMax(
    HandleNewDailyDM,
    HandleNewDailyMax,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )    

@dataclass(frozen=True)
class HandleNewDailyDMed(
    HandleNewDailyDM,
    HandleNewDailyMed,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

@dataclass(frozen=True)
class HandleNewDailyDMid(
    HandleNewDailyDM,
    HandleNewDailyMid,
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedMid,
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
class HandleNewDailyDMin(
    HandleNewDailyDM,
    HandleNewDailyMin,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )


@dataclass(frozen=True)
class HandleNewDailyDPLast(
    HandleNewDailyDM,
    HandleNewDailyPLast,
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
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
class HandleNewDailyDQile(
    HandleNewDailyDM,
    HandleNewDailyQile,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedQile,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )
    

@dataclass(frozen=True)
class HandleNewDailyDRank(
    HandleNewDailyDM,
    HandleNewDailyRank,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    
@dataclass(frozen=True)
class HandleNewDailyDRms(
    HandleNewDailyDM,
    HandleNewDailyRms,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )


    
@dataclass(frozen=True)
class HandleNewDailyDRng(
    HandleNewDailyDM,
    HandleNewDailyRng,
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedMRng,
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
class HandleNewDailyDSDev(
    HandleNewDailyDM,
    HandleNewDailySDev,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    
@dataclass(frozen=True)
class HandleNewDailyDSkew(
    HandleNewDailyDM,
    HandleNewDailySkew,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

    

@dataclass(frozen=True)
class HandleNewDailyDVar(
    HandleNewDailyDM,
    HandleNewDailyVar,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyChunkedM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

#  ------------------

__all__ = [
    "HandleNewDailyDM",
    "HandleNewDailyDMA",
    "HandleNewDailyDKurt",
    "HandleNewDailyDMax",
    "HandleNewDailyDMed",
    "HandleNewDailyDMid",
    "HandleNewDailyDMin",
    "HandleNewDailyDPLast",
    "HandleNewDailyDQile",
    "HandleNewDailyDRank",
    "HandleNewDailyDRms",
    "HandleNewDailyDRng",
    "HandleNewDailyDSDev",
    "HandleNewDailyDSkew",
    "HandleNewDailyDVar",
]