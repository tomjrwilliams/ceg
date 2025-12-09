
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
class DailyEMParams(DailyParams):
    chunk: int = 32
    # TODO: db?

    span: float | None = None
    alpha: float | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.


@dataclass(frozen=True)
class DailyEM(DailyChunked):
    parent: Object | str
    # tags: DailyTags
    
    params: DailyEMParams

    version: int = 0
    type: Literal["DailyEM"] = "DailyEM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "EM"
    # NOTE: to override
    
    def calc_alpha(self) -> float:
        res = (
            self.params.alpha
            if self.params.alpha is not None
            else 2 / (self.params.span + 1)
            if self.params.span is not None
            else None
        )
        assert res is not None, self
        return res
    
    @classmethod
    def params_suffix(
        cls,
        span: float | None = None,
        alpha: float | None = None,
    ):
        i = apis.t.only_index_not_none(
            span, alpha, strict=True
        )
        if i == 0:
            assert span is not None, span
            assert span >= 1, span
            params = f"{span}"
        elif i == 1:
            assert alpha is not None, alpha
            assert 0 < alpha < 1, alpha
            params = f"{alpha}"
        else:
            raise ValueError((span, alpha))
        return f"{cls.SUFFIX}-" + params
    
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        span: float | None=None,
        alpha: float | None=None,
    ) -> DailyEM: ...

    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ],
        span: float | None=None,
        alpha: float | None=None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DailyEM: ...

    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ] | None = None,
        span: float | None=None,
        alpha: float | None=None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DailyEM:
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                span=span,
                alpha=alpha,
            )),
            parent=parent,
            params=DailyEMParams(
                span=span,
                alpha=alpha,
            ),
        )
    
#  ------------------

@dataclass(frozen=True)
class HandleNewDailyEM(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    pass

#  ------------------

@dataclass(frozen=True)
class DailyEMA(DailyEM):

    version: int = 0
    type: Literal["DailyEMA"] = "DailyEMA"

    SUFFIX: ClassVar[str] = "ema"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyEMASnap"] = "DailyEMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyEMA:
        return HandleNewDailyEMA()

@dataclass(frozen=True)
class HandleNewDailyEMA(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).ewm_mean(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )
    
@dataclass(frozen=True)
class DailyEMax(DailyEM):

    version: int = 0
    type: Literal["DailyEMax"] = "DailyEMax"

    SUFFIX: ClassVar[str] = "emax"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyEMaxSnap"] = "DailyEMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyEMax:
        return HandleNewDailyEMax()
    
@dataclass(frozen=True)
class HandleNewDailyEMax(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_max(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            VALUE: res
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

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

    version: int = 0
    type: Literal["DailyEMid"] = "DailyEMid"

    SUFFIX: ClassVar[str] = "emid"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyEMidSnap"] = "DailyEMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyEMid:
        return HandleNewDailyEMid()


@dataclass(frozen=True)
class HandleNewDailyEMid(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        mx = calc_max(v, is_null, alpha=obj.calc_alpha())
        mn = calc_min(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            "max": mx,
            "min": mn,
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(
                (pl.col("max") + pl.col("min")) / 2
            )
            .alias(VALUE)
        )
    
@dataclass(frozen=True)
class DailyEMin(DailyEM):

    version: int = 0
    type: Literal["DailyEMin"] = "DailyEMin"

    SUFFIX: ClassVar[str] = "emin"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyEMinSnap"] = "DailyEMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyEMin:
        return HandleNewDailyEMin()
    
@dataclass(frozen=True)
class HandleNewDailyEMin(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_min(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            VALUE: res
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(VALUE)
            .alias(VALUE)
        )

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
class DailyEPLast(DailyEM):

    version: int = 0
    type: Literal["DailyEPLast"] = "DailyEPLast"

    SUFFIX: ClassVar[str] = "eplast"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyEPLastSnap"] = "DailyEPLastSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyEPLast:
        return HandleNewDailyEPLast()
    
@dataclass(frozen=True)
class HandleNewDailyEPLast(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc_p_last(
            v,
            v,
            is_null,
            alpha=obj.calc_alpha(),
            l=100,
            norm = False, # TODO: parametrise in path?
        )

        return pl.DataFrame({
            "OLD": s,
            VALUE: res
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
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
    alpha: float,
    l: int,
    norm: bool,
):
    n: int = len(v)
    acc: float = 0
    res = np.zeros(n, dtype=v.dtype)
    ws = np.zeros(l, dtype=v.dtype)

    j = 0

    alpha_l = alpha ** l
    while j < l:
        ws[j] = (
            (1 - alpha) / (1 - alpha_l)
        ) * (alpha ** j)
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
                        else 0.5 if v[ij] == loc[i]
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

    version: int = 0
    type: Literal["DailyERms"] = "DailyERms"

    SUFFIX: ClassVar[str] = "erms"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyERmsSnap"] = "DailyERmsSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyERms:
        return HandleNewDailyERms()

@dataclass(frozen=True)
class HandleNewDailyERms(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).pow(2).ewm_mean(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            ).sqrt()
        )

@dataclass(frozen=True)
class DailyERng(DailyEM):

    version: int = 0
    type: Literal["DailyERng"] = "DailyERng"

    SUFFIX: ClassVar[str] = "erng"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyERngSnap"] = "DailyERngSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyERng:
        return HandleNewDailyERng()

@dataclass(frozen=True)
class HandleNewDailyERng(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.DataFrame:
        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        mx = calc_max(v, is_null, alpha=obj.calc_alpha())
        mn = calc_min(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "OLD": s,
            "max": mx,
            "min": mn,
        }).with_columns(
            pl.when(pl.col("OLD").is_null())
            .then(None)
            .otherwise(
                pl.col("max") - pl.col("min")
            )
            .alias(VALUE)
        )
@dataclass(frozen=True)
class DailyESDev(DailyEM):

    version: int = 0
    type: Literal["DailyESDev"] = "DailyESDev"

    SUFFIX: ClassVar[str] = "esdev"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyESDevSnap"] = "DailyESDevSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyESDev:
        return HandleNewDailyESDev()

@dataclass(frozen=True)
class HandleNewDailyESDev(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).ewm_std(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )

@dataclass(frozen=True)
class DailyEVar(DailyEM):

    version: int = 0
    type: Literal["DailyEVar"] = "DailyEVar"

    SUFFIX: ClassVar[str] = "evar"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyEVarSnap"] = "DailyEVarSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyEVar:
        return HandleNewDailyEVar()
    
@dataclass(frozen=True)
class HandleNewDailyEVar(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        return (
            pl.col(VALUE).ewm_var(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )

@dataclass(frozen=True)
class DailyEZ(DailyEM):

    version: int = 0
    type: Literal["DailyEZ"] = "DailyEZ"

    SUFFIX: ClassVar[str] = "ez"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyEZSnap"] = "DailyEZSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyEZ:
        return HandleNewDailyEZ()
    
@dataclass(frozen=True)
class HandleNewDailyEZ(HandleNewDailyEM):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyEM,
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
        obj: DailyEM,
        domain: Domain,
    ) -> pl.Expr:
        params = obj.params
        mu = (
            pl.col(VALUE).ewm_mean(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )
        sig = (
            pl.col(VALUE)
            .forward_fill()
            .diff()
            .pipe(apis.pl.null_mask, pl.col(VALUE))
            .ewm_std(
                span=params.span,
                alpha=params.alpha,
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )
        )
        return (
            (pl.col(VALUE) - mu) / sig
        ).fill_nan(None)

@dataclass(frozen=True)
class DailyERZParams(DailyEMParams):
    chunk: int = 32
    # TODO: db?

    span: float | None = None
    alpha: float | None = None
    width: float = 0.5
    symmetric: bool = True
    rescale: bool = False

@dataclass(frozen=True)
class DailyERZ(DailyEM):

    params: DailyERZParams

    version: int = 0
    type: Literal["DailyERZ"] = "DailyERZ"

    SUFFIX: ClassVar[str] = "erz"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyERZSnap"] = "DailyERZSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewDailyERZ:
        return HandleNewDailyERZ()
    
    @classmethod
    def params_suffix(
        cls,
        span: float | None=None,
        alpha: float | None=None,
        width: float | None = None,
        symmetric: bool = True,
        rescale: bool = False,
    ):
        i = apis.t.only_index_not_none(
            span, alpha, strict=True
        )
        if i == 0:
            assert span is not None, span
            assert span >= 1, span
            params = f"{span}"
        elif i == 1:
            assert alpha is not None, alpha
            assert 0 < alpha < 1, alpha
            params = f"{alpha}"
        else:
            raise ValueError((span, alpha))
        return f"{cls.SUFFIX}-" + params + (
            ""
            if width is None
            else f"-{width}"
        ) + (
            "-sym"
            if symmetric
            else ""
        ) + (
            "-norm"
            if rescale
            else ""
        )
    
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        span: float | None=None,
        alpha: float | None=None,
        width: float = 0.5,
        symmetric: bool = False,
        rescale: bool = False,
    ) -> DailyEM: ...

    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ],
        span: float | None=None,
        alpha: float | None=None,
        width: float = 0.5,
        symmetric: bool = False,
        rescale: bool = False,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DailyEM: ...

    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ] | None = None,
        span: float | None=None,
        alpha: float | None=None,
        width: float = 0.5,
        symmetric: bool = False,
        rescale: bool = False,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DailyEM:
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                span=span,
                alpha=alpha,
                width=width,
                symmetric=symmetric,
                rescale=rescale,
            )),
            parent=parent,
            params=DailyERZParams(
                span=span,
                alpha=alpha,
                width=width,
                symmetric=symmetric,
                rescale=rescale,
            ),
        )
    
@dataclass(frozen=True)
class HandleNewDailyERZ(HandleNewDailyEM):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyERZ,
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
        obj: DailyERZ,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params

        assert params.width == 0.5, params

        if params.rescale:
            scale = apis.stats.iqr_ratio(
                width=params.width
            )
        else:
            scale = 1

        s = df.get_column(VALUE)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()

        mx = calc_max(v, is_null, alpha=obj.calc_alpha())
        mn = calc_min(v, is_null, alpha=obj.calc_alpha())

        v_ex = pl.col("OLD")

        df = pl.DataFrame({
            "OLD": s,
            "max": mx,
            "min": mn,
        }).with_columns(
            (pl.col("max") - pl.col("min"))
            .mul(1/2)
            .mul(scale)
            .alias("rng")
        )

        if params.symmetric:
            mid = (pl.col("max") + pl.col("min")) / 2
        else:
            mid = v_ex.ewm_mean(
                alpha=obj.calc_alpha(),
                min_samples=1, # TODO: or more?
                ignore_nulls=True,
            )

        df = df.with_columns(
            ((v_ex - mid) / pl.col("rng"))
            .alias(VALUE)
        )
        return df.with_columns(
            pl.when(v_ex.is_null())
            .then(None)
            .otherwise(pl.col(VALUE))
            .fill_nan(None)
            .alias(VALUE)
        )

__all__ = [
    "P", "O",
    "DailyEMParams",
    "DailyEM",
    "HandleNewDailyEM",
    "HandleNewDailyEMA",
    "DailyEMA",
    "HandleNewDailyEMax",
    "DailyEMax",
    "calc_max",
    "HandleNewDailyEMid",
    "DailyEMid",
    "HandleNewDailyEMin",
    "DailyEMin",
    "calc_min",
    "HandleNewDailyEPLast",
    "DailyEPLast",
    "calc_p_last",
    "HandleNewDailyERms",
    "DailyERms",
    "HandleNewDailyERng",
    "DailyERng",
    "HandleNewDailyESDev",
    "DailyESDev",
    "HandleNewDailyEVar",
    "DailyEVar",
    "HandleNewDailyEZ",
    "DailyEZ",
    "HandleNewDailyERZ",
    "DailyERZ",
]