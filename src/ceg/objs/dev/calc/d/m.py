
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

P = ParamSpec("P")
O = TypeVar("O", bound=Object)

#  ------------------

@dataclass(frozen=True)
class DailyMParams(DailyParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.


@dataclass(frozen=True)
class DailyM(DailyChunked):
    parent: Object | str
    # tags: DailyTags
    params: DailyMParams

    version: int = 0
    type: Literal["DailyM"] = "DailyM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "M"
    # NOTE: to override
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
    ):
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params
    
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        window: int | None = None,
    ): ...
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ],
        window: int | None,
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
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=window,
            )),
            parent=parent,
            params=DailyMParams(
                window=window,
            ),
        )

#  ------------------

@dataclass(frozen=True)
class HandleNewDailyM(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class DailyMA(DailyM):
    version: int = 0
    type: Literal["DailyMA"] = "DailyMA"

    SUFFIX: ClassVar[str] = "ma"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyMASnap"] = "DailyMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyMA:
        return HandleNewDailyMA()

@dataclass(frozen=True)
class HandleNewDailyMA(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).rolling_mean(
                len(df) 
                if params.window is None 
                else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    
@dataclass(frozen=True)
class DailyKurt(DailyM):
    version: int = 0
    type: Literal["DailyKurt"] = "DailyKurt"

    SUFFIX: ClassVar[str] = "kurt"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyKurtSnap"] = "DailyKurtSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyKurt:
        return HandleNewDailyKurt()
@dataclass(frozen=True)
class HandleNewDailyKurt(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).rolling_kurtosis(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

@dataclass(frozen=True)
class DailyMax(DailyM):
    version: int = 0
    type: Literal["DailyMax"] = "DailyMax"

    SUFFIX: ClassVar[str] = "max"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyMaxSnap"] = "DailyMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyMax:
        return HandleNewDailyMax()
@dataclass(frozen=True)
class HandleNewDailyMax(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_max(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    

@dataclass(frozen=True)
class DailyMed(DailyM):
    version: int = 0
    type: Literal["DailyMed"] = "DailyMed"

    SUFFIX: ClassVar[str] = "med"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyMedSnap"] = "DailyMedSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyMed:
        return HandleNewDailyMed()
@dataclass(frozen=True)
class HandleNewDailyMed(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_median(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

@dataclass(frozen=True)
class DailyMid(DailyM):

    params: DailyMidParams

    version: int = 0
    type: Literal["DailyMid"] = "DailyMid"

    SUFFIX: ClassVar[str] = "mid"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyMidSnap"] = "DailyMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyMid:
        return HandleNewDailyMid()

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
        ],
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
class DailyMidParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    width: float | None = None

    
@dataclass(frozen=True)
class HandleNewDailyMid(HandleNewDailyM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyMid,
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
        obj: DailyMid,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
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
    

@dataclass(frozen=True)
class DailyMin(DailyM):
    version: int = 0
    type: Literal["DailyMin"] = "DailyMin"

    SUFFIX: ClassVar[str] = "min"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyMinSnap"] = "DailyMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyMin:
        return HandleNewDailyMin()

@dataclass(frozen=True)
class HandleNewDailyMin(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        
        return(
            pl.col(VALUE).rolling_min(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

@dataclass(frozen=True)
class DailyPLast(DailyM):
    version: int = 0
    type: Literal["DailyPLast"] = "DailyPLast"

    SUFFIX: ClassVar[str] = "plast"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyPLastSnap"] = "DailyPLastSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyPLast:
        return HandleNewDailyPLast()


@dataclass(frozen=True)
class HandleNewDailyPLast(HandleNewDailyM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_p_last(
            v,
            v,
            is_null,
            l=len(df) if params.window is None else params.window,
            norm = False, # TODO: parametrise in path?
        )

        return pl.DataFrame({
            "mask": s,
            VALUE: res
        }).select(
            pl.when(pl.col("mask").is_null())
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
                        else 0.5 if v[ij] == loc[i]
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
class DailyQile(DailyM):

    params: DailyQileParams

    version: int = 0
    type: Literal["DailyQile"] = "DailyQile"

    SUFFIX: ClassVar[str] = "qile"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyQileSnap"] = "DailyQileSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyQile:
        return HandleNewDailyQile()
    
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
class DailyQileParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    quantile: float | None = None


@dataclass(frozen=True)
class HandleNewDailyQile(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyQile,
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
        obj: DailyQile,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        assert params.quantile is not None, obj
        
        return(
            pl.col(VALUE).rolling_quantile(
                params.quantile,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    

@dataclass(frozen=True)
class DailyRank(DailyM):
    version: int = 0
    type: Literal["DailyRank"] = "DailyRank"

    SUFFIX: ClassVar[str] = "rank"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyRankSnap"] = "DailyRankSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyRank:
        return HandleNewDailyRank()
@dataclass(frozen=True)
class HandleNewDailyRank(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_rank(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

    
@dataclass(frozen=True)
class DailyRms(DailyM):
    version: int = 0
    type: Literal["DailyRms"] = "DailyRms"

    SUFFIX: ClassVar[str] = "rms"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyRmsSnap"] = "DailyRmsSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyRms:
        return HandleNewDailyRms()

@dataclass(frozen=True)
class HandleNewDailyRms(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).pow(2).rolling_mean(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            ).sqrt()
        )

@dataclass(frozen=True)
class DailyRng(DailyM):

    params: DailyRngParams

    version: int = 0
    type: Literal["DailyRng"] = "DailyRng"

    SUFFIX: ClassVar[str] = "rng"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyRngSnap"] = "DailyRngSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyRng:
        return HandleNewDailyRng()
    
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
        ],
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
            )),
            parent=parent,
            params=DailyRngParams(
                window=window,
                width=width,
            ),
        )
    
    
@dataclass(frozen=True)
class DailyRngParams(DailyMParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    width: float | None = None


@dataclass(frozen=True)
class HandleNewDailyRng(HandleNewDailyM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyRng,
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
        obj: DailyRng,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        e = pl.col(VALUE)
        u = (
            e.rolling_max(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 + params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
            if params.width is None
            else e.rolling_quantile(
                0.5 - params.width,
                window_size=len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
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

@dataclass(frozen=True)
class DailySDev(DailyM):
    version: int = 0
    type: Literal["DailySDev"] = "DailySDev"

    SUFFIX: ClassVar[str] = "sdev"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailySDevSnap"] = "DailySDevSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailySDev:
        return HandleNewDailySDev()
@dataclass(frozen=True)
class HandleNewDailySDev(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_std(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
@dataclass(frozen=True)
class DailySkew(DailyM):
    version: int = 0
    type: Literal["DailySkew"] = "DailySkew"

    SUFFIX: ClassVar[str] = "skew"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailySkewSnap"] = "DailySkewSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailySkew:
        return HandleNewDailySkew()

    
@dataclass(frozen=True)
class HandleNewDailySkew(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_skew(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )
    

@dataclass(frozen=True)
class DailyVar(DailyM):
    version: int = 0
    type: Literal["DailyVar"] = "DailyVar"

    SUFFIX: ClassVar[str] = "var"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyVarSnap"] = "DailyVarSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyVar:
        return HandleNewDailyVar()
    
@dataclass(frozen=True)
class HandleNewDailyVar(HandleNewDailyM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyM,
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
        obj: DailyM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return(
            pl.col(VALUE).rolling_var(
                len(df) if params.window is None else params.window,
                min_samples=1, # TODO: or more?
            )
        )

#  ------------------

__all__ = [
    "P", "O",
    "DailyMParams",
    "DailyM",
    "HandleNewDailyM",
    "HandleNewDailyMA",
    "DailyMA",
    "HandleNewDailyKurt",
    "DailyKurt",
    "HandleNewDailyMax",
    "DailyMax",
    "HandleNewDailyMed",
    "DailyMed",
    "DailyMidParams",
    "DailyMid",
    "HandleNewDailyMid",
    "DailyMid",
    "HandleNewDailyMin",
    "DailyMin",
    "HandleNewDailyPLast",
    "DailyPLast",
    "calc_p_last",
    "DailyQileParams",
    "DailyQile",
    "HandleNewDailyQile",
    "DailyQile",
    "HandleNewDailyRank",
    "DailyRank",
    "HandleNewDailyRms",
    "DailyRms",
    "DailyRngParams",
    "DailyRng",
    "HandleNewDailyRng",
    "DailyRng",
    "HandleNewDailySDev",
    "DailySDev",
    "HandleNewDailySkew",
    "DailySkew",
    "HandleNewDailyVar",
    "DailyVar",
]