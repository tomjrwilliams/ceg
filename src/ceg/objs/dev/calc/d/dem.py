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
    HandleNewValueFromPl,
    HandleNewDailyFromDailyParent,
):
    pass


#  ------------------


@dataclass(frozen=True)
class DailyDEMA(DailyDEM):
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

    version: int = 0
    type: Literal["DailyDEMA"] = "DailyDEMA"

    SUFFIX: ClassVar[str] = "dema"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDEMASnap"] = "DailyDEMASnap"

    def handler_object_new(self) -> HandleNewDailyDEMA:
        return HandleNewDailyDEMA()


@dataclass(frozen=True)
class HandleNewDailyDEMA(
    HandleNewDailyDEM, HandleNewDailyEMA
):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )


@dataclass(frozen=True)
class DailyDEMax(DailyDEM):
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

    version: int = 0
    type: Literal["DailyDEMax"] = "DailyDEMax"

    SUFFIX: ClassVar[str] = "demax"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDEMaxSnap"] = "DailyDEMaxSnap"

    def handler_object_new(self) -> HandleNewDailyDEMax:
        return HandleNewDailyDEMax()


@dataclass(frozen=True)
class HandleNewDailyDEMax(
    HandleNewDailyDEM, HandleNewDailyEMax
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
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
class DailyDEMid(DailyDEM):
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

    version: int = 0
    type: Literal["DailyDEMid"] = "DailyDEMid"

    SUFFIX: ClassVar[str] = "demid"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDEMidSnap"] = "DailyDEMidSnap"

    def handler_object_new(self) -> HandleNewDailyDEMid:
        return HandleNewDailyDEMid()


@dataclass(frozen=True)
class HandleNewDailyDEMid(
    HandleNewDailyDEM, HandleNewDailyEMid
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
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
class DailyDEMin(DailyDEM):
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

    version: int = 0
    type: Literal["DailyDEMin"] = "DailyDEMin"

    SUFFIX: ClassVar[str] = "demin"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDEMinSnap"] = "DailyDEMinSnap"

    def handler_object_new(self) -> HandleNewDailyDEMin:
        return HandleNewDailyDEMin()


@dataclass(frozen=True)
class HandleNewDailyDEMin(
    HandleNewDailyDEM, HandleNewDailyEMin
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.DataFrame:
        df = df.with_columns(pl.col(VALUE).alias("OLD"))
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
class DailyDEPLast(DailyDEM):
    version: int = 0
    type: Literal["DailyDEPLast"] = "DailyDEPLast"

    SUFFIX: ClassVar[str] = "deplast"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDEPLastSnap"] = (
            "DailyDEPLastSnap"
        )

    def handler_object_new(self) -> HandleNewDailyDEPLast:
        return HandleNewDailyDEPLast()


@dataclass(frozen=True)
class HandleNewDailyDEPLast(
    HandleNewDailyDEM, HandleNewDailyEPLast
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
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
class DailyDERms(DailyDEM):
    version: int = 0
    type: Literal["DailyDERms"] = "DailyDERms"

    SUFFIX: ClassVar[str] = "derms"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDERmsSnap"] = "DailyDERmsSnap"

    def handler_object_new(self) -> HandleNewDailyDERms:
        return HandleNewDailyDERms()


@dataclass(frozen=True)
class HandleNewDailyDERms(
    HandleNewDailyDEM, HandleNewDailyERms
):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )


@dataclass(frozen=True)
class DailyDERng(DailyDEM):
    version: int = 0
    type: Literal["DailyDERng"] = "DailyDERng"

    SUFFIX: ClassVar[str] = "derng"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDERngSnap"] = "DailyDERngSnap"

    def handler_object_new(self) -> HandleNewDailyDERng:
        return HandleNewDailyDERng()


@dataclass(frozen=True)
class HandleNewDailyDERng(
    HandleNewDailyDEM, HandleNewDailyERng
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
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
class DailyDEHwm(DailyDEM):
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

    version: int = 0
    type: Literal["DailyDEHwm"] = "DailyDEHwm"

    SUFFIX: ClassVar[str] = "dehwm"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDEHwmSnap"] = "DailyDEHwmSnap"

    def handler_object_new(self) -> HandleNewDailyDEHwm:
        return HandleNewDailyDEHwm()


@dataclass(frozen=True)
class HandleNewDailyDEHwm(
    HandleNewDailyDEM, HandleNewDailyEHwm
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
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
class DailyDELwm(DailyDEM):
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

    version: int = 0
    type: Literal["DailyDELwm"] = "DailyDELwm"

    SUFFIX: ClassVar[str] = "delwm"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDELwmSnap"] = "DailyDELwmSnap"

    def handler_object_new(self) -> HandleNewDailyDELwm:
        return HandleNewDailyDELwm()


@dataclass(frozen=True)
class HandleNewDailyDELwm(
    HandleNewDailyDEM, HandleNewDailyELwm
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.DataFrame:

        df = df.with_columns(pl.col(VALUE).alias("OLD"))
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
class DailyDESDev(DailyDEM):
    version: int = 0
    type: Literal["DailyDESDev"] = "DailyDESDev"

    SUFFIX: ClassVar[str] = "desdev"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDESDevSnap"] = "DailyDESDevSnap"

    def handler_object_new(self) -> HandleNewDailyDESDev:
        return HandleNewDailyDESDev()


@dataclass(frozen=True)
class HandleNewDailyDESDev(
    HandleNewDailyDEM, HandleNewDailyESDev
):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
        )


@dataclass(frozen=True)
class DailyDEVar(DailyDEM):
    version: int = 0
    type: Literal["DailyDEVar"] = "DailyDEVar"

    SUFFIX: ClassVar[str] = "devar"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDEVarSnap"] = "DailyDEVarSnap"

    def handler_object_new(self) -> HandleNewDailyDEVar:
        return HandleNewDailyDEVar()


@dataclass(frozen=True)
class HandleNewDailyDEVar(
    HandleNewDailyDEM, HandleNewDailyEVar
):
    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyEM,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
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
    "DailyDEPLast",
    "HandleNewDailyDERms",
    "DailyDERms",
    "HandleNewDailyDERng",
    "DailyDERng",
    "HandleNewDailyDEHwm",
    "DailyDEHwm",
    "HandleNewDailyDELwm",
    "DailyDELwm",
    "HandleNewDailyDESDev",
    "DailyDESDev",
    "HandleNewDailyDEVar",
    "DailyDEVar",
]
