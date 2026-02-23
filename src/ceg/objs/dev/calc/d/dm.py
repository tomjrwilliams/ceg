
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

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyDM(
    HandleNewValueFromPl,
    HandleNewDailyFromDailyParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class DailyDMA(DailyDM):
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
    >>> obj.value_key_val().get(obj.path).tail(8)
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
    version: int = 0
    type: Literal["DailyDMA"] = "DailyDMA"

    SUFFIX: ClassVar[str] = "dma"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDMASnap"] = "DailyDMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDMA:
        return HandleNewDailyDMA()
@dataclass(frozen=True)
class HandleNewDailyDMA(
    HandleNewDailyDM,
    HandleNewDailyMA,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )
    
@dataclass(frozen=True)
class HandleNewDailyDKurt(
    HandleNewDailyDM,
    HandleNewDailyKurt,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )

@dataclass(frozen=True)
class DailyDMax(DailyDM):
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
    >>> obj.value_key_val().get(obj.path).tail(8)
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
    version: int = 0
    type: Literal["DailyDMax"] = "DailyDMax"

    SUFFIX: ClassVar[str] = "dmax"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDMaxSnap"] = "DailyDMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDMax:
        return HandleNewDailyDMax()


@dataclass(frozen=True)
class HandleNewDailyDMax(
    HandleNewDailyDM,
    HandleNewDailyMax,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )    

@dataclass(frozen=True)
class DailyDMed(DailyDM):
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
    >>> obj.value_key_val().get(obj.path).tail(8)
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

    SUFFIX: ClassVar[str] = "dmed"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDMedSnap"] = "DailyDMedSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDMed:
        return HandleNewDailyDMed()

@dataclass(frozen=True)
class HandleNewDailyDMed(
    HandleNewDailyDM,
    HandleNewDailyMed,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )

@dataclass(frozen=True)
class DailyDMid(DailyDM):
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
    >>> obj.value_key_val().get(obj.path).tail(8)
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
    params: DailyMidParams

    version: int = 0
    type: Literal["DailyDMid"] = "DailyDMid"

    SUFFIX: ClassVar[str] = "dmid"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDMidSnap"] = "DailyDMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDMid:
        return HandleNewDailyDMid()
    
    @classmethod
    def new(
        cls,
        parent: Daily,
        window: int | None = None,
        width: float | None = None,
        project: Project | None = None,
    ):
        params=DailyMidParams(
            window=window,
            width=width,
        )
        assert cls.SUFFIX != "M", cls
        path = parent.extend_path(
            params.build_path(cls)
        )
        if project is not None:
            path = project.extend_path(path)
        return cls(
            path=path,
            parent=parent,
            params=params,
            project=project,
        )

@dataclass(frozen=True)
class HandleNewDailyDMid(
    HandleNewDailyDM,
    HandleNewDailyMid,
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyMid,
    ) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
    

@dataclass(frozen=True)
class DailyDMin(DailyDM):
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
    >>> obj.value_key_val().get(obj.path).tail(8)
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
    version: int = 0
    type: Literal["DailyDMin"] = "DailyDMin"

    SUFFIX: ClassVar[str] = "dmin"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDMinSnap"] = "DailyDMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDMin:
        return HandleNewDailyDMin()

@dataclass(frozen=True)
class HandleNewDailyDMin(
    HandleNewDailyDM,
    HandleNewDailyMin,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )


@dataclass(frozen=True)
class HandleNewDailyDPLast(
    HandleNewDailyDM,
    HandleNewDailyPLast,
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
        )
        e = pl.col("OLD") - pl.col(VALUE)
        return df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
    
@dataclass(frozen=True)
class DailyDQile(DailyDM):
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
    >>> obj.value_key_val().get(obj.path).tail(8)
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

    params: DailyQileParams

    version: int = 0
    type: Literal["DailyDQile"] = "DailyDQile"

    SUFFIX: ClassVar[str] = "dqile"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDQileSnap"] = "DailyDQileSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDQile:
        return HandleNewDailyDQile()
    
    @classmethod
    def new(
        cls,
        parent: Daily,
        window: int | None = None,
        quantile: float | None = None,
        project: Project | None = None,
    ):
        params=DailyQileParams(
            window=window,
            quantile=quantile,
        )
        assert cls.SUFFIX != "M", cls
        path = parent.extend_path(
            params.build_path(cls)
        )
        if project is not None:
            path = project.extend_path(path)
        return cls(
            path=path,
            parent=parent,
            params=params,
            project=project,
        )

@dataclass(frozen=True)
class HandleNewDailyDQile(
    HandleNewDailyDM,
    HandleNewDailyQile,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyQile,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )
    

@dataclass(frozen=True)
class HandleNewDailyDRank(
    HandleNewDailyDM,
    HandleNewDailyRank,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )

    
@dataclass(frozen=True)
class HandleNewDailyDRms(
    HandleNewDailyDM,
    HandleNewDailyRms,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )


    
@dataclass(frozen=True)
class HandleNewDailyDRng(
    HandleNewDailyDM,
    HandleNewDailyRng,
):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyRng,
    ) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(VALUE).alias("OLD")
        )
        df = self._calc_df(
            df,
            event,
            obj,
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
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )

    
@dataclass(frozen=True)
class HandleNewDailyDSkew(
    HandleNewDailyDM,
    HandleNewDailySkew,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )

    

@dataclass(frozen=True)
class HandleNewDailyDVar(
    HandleNewDailyDM,
    HandleNewDailyVar,
):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )

#  ------------------

__all__ = [
    # "DailyDMParams",
    "DailyDM",
    "HandleNewDailyDM",
    "HandleNewDailyDMA",
    "DailyDMA",
    "HandleNewDailyDKurt",
    "HandleNewDailyDMax",
    "DailyDMax",
    "HandleNewDailyDMed",
    "DailyDMed",
    "HandleNewDailyDMid",
    "DailyDMid",
    "HandleNewDailyDMin",
    "DailyDMin",
    "HandleNewDailyDPLast",
    "HandleNewDailyDQile",
    "DailyDQile",
    "HandleNewDailyDRank",
    "HandleNewDailyDRms",
    "HandleNewDailyDRng",
    "HandleNewDailyDSDev",
    "HandleNewDailyDSkew",
    "HandleNewDailyDVar",
]