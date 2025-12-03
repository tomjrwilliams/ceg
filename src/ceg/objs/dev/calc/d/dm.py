
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
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class DailyDMA(DailyDM):
    version: int = 0
    type: Literal["DailyDMA"] = "DailyDMA"

    SUFFIX: ClassVar[str] = "dma"

    @dataclass(frozen=True)
    class Snap(DailySnap):
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
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

@dataclass(frozen=True)
class DailyDMax(DailyDM):
    version: int = 0
    type: Literal["DailyDMax"] = "DailyDMax"

    SUFFIX: ClassVar[str] = "dmax"

    @dataclass(frozen=True)
    class Snap(DailySnap):
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
        event: ObjectNew,
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )    

@dataclass(frozen=True)
class DailyDMed(DailyDM):
    version: int = 0
    type: Literal["DailyDMed"] = "DailyDMed"

    SUFFIX: ClassVar[str] = "dmed"

    @dataclass(frozen=True)
    class Snap(DailySnap):
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
        event: ObjectNew,
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        return pl.col(VALUE) - self._calc_expr(
            df,
            event,
            obj,
            domain,
        )

@dataclass(frozen=True)
class DailyDMid(DailyDM):
    params: DailyMidParams

    version: int = 0
    type: Literal["DailyDMid"] = "DailyDMid"

    SUFFIX: ClassVar[str] = "dmid"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyDMidSnap"] = "DailyDMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDMid:
        return HandleNewDailyDMid()
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        width: float | None = None,
    ):
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params + (
            ""
            if width is None
            else f"-{width}"
        )
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        window: int | None = None,
        width: float | None = None,
    ): ...
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ] | None = None,
        window: int | None = None,
        width: float | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ): ...
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ] | None = None,
        window: int | None = None,
        width: float | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=window,
                width=width,
            )),
            parent=parent,
            params=DailyMidParams(
                window=window,
                width=width,

            ),
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
        obj: DailyMid,
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
class DailyDMin(DailyDM):
    version: int = 0
    type: Literal["DailyDMin"] = "DailyDMin"

    SUFFIX: ClassVar[str] = "dmin"

    @dataclass(frozen=True)
    class Snap(DailySnap):
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
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
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
class DailyDQile(DailyDM):

    params: DailyQileParams

    version: int = 0
    type: Literal["DailyDQile"] = "DailyDQile"

    SUFFIX: ClassVar[str] = "dqile"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyDQileSnap"] = "DailyDQileSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyDQile:
        return HandleNewDailyDQile()
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        quantile: float | None = None,
    ):
        assert quantile is not None, (cls, quantile)
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params + f"-{quantile}"
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        window: int | None = None,
        quantile: float | None = None,
    ): ...
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ],
        window: int | None = None,
        quantile: float | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ): ...
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ] | None = None,
        window: int | None = None,
        quantile: float | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=window,
                quantile=quantile,
            )),
            parent=parent,
            params=DailyQileParams(
                window=window,
                quantile=quantile,
            ),
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
        obj: DailyQile,
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
        obj: DailyM,
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
        obj: DailyM,
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
        obj: DailyRng,
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
        obj: DailyM,
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
        obj: DailyM,
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
        obj: DailyM,
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