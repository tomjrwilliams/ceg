
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

from ....calc import *

from ...root import *
from ..bars import *
from ..calc import *

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
class FutDailyEOsc(
    FutDailyEM,
    HasFutDailyGrandParent,
):
    """FutDailyEOsc(UseRootConfigs):\n
    parent: FutDaily\n
    symbol: str\n
    field: str\n
    adjust: str\n
    fx: str | None = None # ccy to convert to\n
    source: str | None = None\n
    """
    parent: FutDailyOrdinal | FutDailyRx | str
    # TODO: or contract
    # TODO: or rx thereof

    params: DailyEMOscParams

    version: int = 0
    type: Literal["FutDailyEOsc"] = "FutDailyEOsc"

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
        parent: FutDailyOrdinal | FutDailyRx | EngineRoot,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        # NOTE: just for contracts?
        source: str | None = None,
        fx: str | None = None,
        # TODO: expect, to decide on ordinal
        # --
        fast: float | None = None,
        slow: int | None = None,
        vol: int | None = None,
        activation: str | None = None,
    ):
        if isinstance(parent, EngineRoot):
            assert isinstance(symbol, str), symbol
            if ident is not None:
                assert isinstance(ident, int), ident
            parent = FutDailyOrdinal.new(
                parent,
                symbol=symbol,
                ident=ident,
                field=field,
                adjust=adjust,
                source=source,
                fx=fx,
            )
        span = None
        alpha = None
        if fast is None:
            pass
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
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEOscSnap"] = "FutDailyEOscSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEOsc:
        return HandleNewFutDailyEOsc()

from ......apis.pl import null_mask

@dataclass(frozen=True)
class HandleNewFutDailyEOsc(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: Event,
        obj: FutDailyEOsc,
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
            if params.span is None
            or params.alpha is None
            or params.span == 1
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
class FutDailyEOscDomain(FutDailyDomain):
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
    >>> obj = FutDailyEOsc.new(ord, fast=4, slow=12, vol=8)
    >>> unit = FutDailyEOsc.new(ord, fast=4, slow=12, vol=8, activation="tanh")
    >>> domain = (
    ...     FutDailyEOscDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .with_obj(unit)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eosc-4-12-8' [f64]
    [
        -0.334995
        null
        null
        0.720977
        0.724295
        -0.084052
        -0.344364
        -1.433379
    ]
    >>> unit.value_key_val(domain).get(unit.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eosc-4-12-8-tanh' [f64]
    [
        -0.323002
        null
        null
        0.617514
        0.619563
        -0.083855
        -0.331368
        -0.892357
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEOscDomain"] = "FutDailyEOscDomain"


        SUFFIX: ClassVar[str] = "eosc"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyEOsc
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEOsc.Snap
        )

# TODO: with_activation on the object itself (return a new verison with path updated, etc.)
