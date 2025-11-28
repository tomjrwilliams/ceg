
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

from ....... import apis
from .......apis import t
from .......apis import kv
from .......engine import *

from ..prelude import *
from ..base.em import *
from ...base.em import *
from ...handlers.em import *

from .......apis.pl import null_mask

@dataclass(frozen=True)
class DailyEMOscParams(DailyEMParams):
    
    chunk: int = 32
    # TODO: db?

    span: float | None = None
    alpha: float | None = None

    slow: int | None = None
    vol: int | None = None
    activation: str | None = None

@dataclass(frozen=True)
class BarDailyEOsc(
    BarDailyEM,
):
    params: DailyEMOscParams

    version: int = 0
    type: Literal["BarDailyEOsc"] = "BarDailyEOsc"

    SUFFIX: ClassVar[str] = "eosc"
    
    @classmethod
    def params_suffix(
        cls,
        span: float | None = None,
        alpha: float | None = None,
        slow: int | None = None,
        vol: int | None = None,
        activation: str | None = None,
    ):
        fast = (
            1 if span is None and alpha is None
            else span if alpha is None
            else alpha if span is None
            else None
        )
        assert fast is not None, fast
        s_slow = "ex" if slow is None else slow
        s_vol = "ex" if vol is None else vol
        res = f"{cls.SUFFIX}-{fast}-{s_slow}-{s_vol}"
        if activation is not None:
            return f"{res}-{activation}"
        return res

    @classmethod
    def new(
        cls,
        parent: (
            EngineRoot
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyStock
            | EqDailyRx
            | IndexDailyPx
            | IndexDailyRx
            | FxDailyPx
            | FxDailyRx
        ),
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        # --
        fast: float | None = None,
        slow: int | None = None,
        vol: int | None = None,
        activation: str | None = None,
        #
        t: (
            None
            | Type[FutDailyOrdinal]
            | Type[FutDailyRx]
            | Type[EqDailyEtf]
            | Type[EqDailyStock]
            | Type[EqDailyRx]
            | Type[IndexDailyPx]
            | Type[IndexDailyRx]
            | Type[FxDailyPx]
            | Type[FxDailyRx]
        ) = None,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            assert isinstance(symbol, str), (cls, t)
            assert adjust is not None, adjust
            parent = t.new(
                parent,
                symbol,
                ident=ident,
                field=field,
                adjust=adjust,
                source=source,
                fx=fx,
            )
        span = None
        alpha = None
        if fast is None:
            raise ValueError(fast)
        elif fast >= 1:
            span = fast
        else:
            alpha = fast
        return cls(
            path=parent.extend_path(cls.params_suffix(
                span=span,
                alpha=alpha,
                slow=slow,
                vol=vol,
                activation=activation
            )),
            parent=parent,
            params=DailyEMOscParams(
                span=span,
                alpha=alpha,
                slow=slow,
                vol=vol,
                activation=activation,
            )
        )

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyEOscSnap"] = "BarDailyEOscSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyEOsc:
        return HandleNewBarDailyEOsc()


@dataclass(frozen=True)
class HandleNewBarDailyEOsc(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: Event,
        obj: BarDailyEOsc,
        domain: Domain,
    ) -> pl.DataFrame:
        params = obj.params
        parent = obj.get_parent(domain)
        if isinstance(parent, FutDailyRx):
            tags = parent.get_parent(domain).tags
            rx = pl.col(VALUE)
            if tags.adjust == "pct":
                px = (1 + rx).cum_prod().pipe(
                    null_mask, rx
                )
            elif tags.adjust == "abs":
                px = rx.cum_sum().pipe(
                    null_mask, rx
                )
            else:
                raise ValueError(obj)
        elif isinstance(parent, FutDailyOrdinal):
            tags = parent.tags
            px = pl.col(VALUE)
            if tags.adjust == "pct":
                rx = px.forward_fill().pct_change().pipe(
                    null_mask, px
                )
            elif tags.adjust == "abs":
                rx = px.forward_fill().diff().pipe(
                    null_mask, px
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
            if (
                params.span is None
                and params.alpha is None
            ) or params.span == 1
            else px.ewm_mean(
                span=params.span, min_samples=2,
                ignore_nulls=True,
            )
            if params.span is not None
            else px.ewm_mean(
                alpha=params.alpha, min_samples=2,
                ignore_nulls=True,
            )
            if params.alpha is not None
            else None
        )
        assert px_fast is not None, obj
        px_slow = (
            px.rolling_mean(
                len(df), min_samples=2,
                # ignore_nulls=True,
            )
            if params.slow is None
            else px.ewm_mean(
                span=params.slow, min_samples=2,
                ignore_nulls=True,
            )
        )
        # TODO: or default to rms?
        rx_vol = (
            rx.rolling_std(
                len(df), min_samples=2,
                # ignore_nulls=True,
            )
            if params.vol is None
            else px.ewm_std(
                span=params.vol, min_samples=2,
                ignore_nulls=True,
            )
        )

        e = (
            ((px_fast - px_slow) / rx_vol)
            .pipe(null_mask, pl.col(VALUE))
        )
        if params.activation is None:
            pass
        elif params.activation == "tanh":
            e = e.tanh()
        else:
            raise ValueError(obj)

        df = df.select(
            e.alias(VALUE)
        )
        return df

@dataclass(frozen=True)
class BarDailyEOscDomain(DailyChunkedDomain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> ord = FutDailyOrdinal.new(
    ...     eng,
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = BarDailyEOsc.new(ord, fast=4, slow=12, vol=8)
    >>> unit = BarDailyEOsc.new(ord, fast=4, slow=12, vol=8, activation="tanh")
    >>> domain = (
    ...     BarDailyEOscDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .with_obj(unit)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eosc-4-12-8' [f64]
    [
        -0.418839
        null
        null
        -0.033027
        0.190841
        0.105384
        -0.02915
        -0.430837
    ]
    >>> unit.value_key_val(domain).get(unit.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eosc-4-12-8-tanh' [f64]
    [
        -0.395952
        null
        null
        -0.033015
        0.188558
        0.104996
        -0.029142
        -0.40602
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyEOscDomain"] = "BarDailyEOscDomain"

        SUFFIX: ClassVar[str] = "eosc"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyEOsc
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyEOsc.Snap
        )

# TODO: with_activation on the object itself (return a new verison with path updated, etc.)

__all__ = [
    "BarDailyEOsc",
    "HandleNewBarDailyEOsc",
    "BarDailyEOscDomain",
]