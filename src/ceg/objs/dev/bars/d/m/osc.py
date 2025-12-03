
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

from ......apis.pl import null_mask

from ...... import apis
from ......apis import t
from ......apis import kv
from ......engine import *

from ..prelude import *

from ....calc.d import *

@dataclass(frozen=True)
class DailyMOscParams(DailyMParams):
    
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    slow: int | None = None
    vol: int | None = None
    activation: str | None = None

@dataclass(frozen=True)
class BarDailyOsc(DailyM):
    parent: (
        str
        | FutDailyOrdinal
        | FutDailyRx
        | EqDailyEtf
        | EqDailyStock
        | EqDailyRx
        | IndexDailyPx
        | IndexDailyRx
        | FxDailyPx
        | FxDailyRx
    )

    params: DailyMOscParams

    version: int = 0
    type: Literal["BarDailyOsc"] = "BarDailyOsc"

    SUFFIX: ClassVar[str] = "osc"
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        slow: int | None = None,
        vol: int | None = None,
        activation: str | None = None,
    ):
        fast = 1 if window is None else window
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
        fast: int | None = None,
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
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=fast,
                slow=slow,
                vol=vol,
                activation=activation
            )),
            parent=parent,
            params=DailyMOscParams(
                window=fast,
                slow=slow,
                vol=vol,
                activation=activation
            ),
        )

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyOscSnap"] = "BarDailyOscSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyOsc:
        return HandleNewBarDailyOsc()

@dataclass(frozen=True)
class HandleNewBarDailyOsc(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: Event,
        obj: BarDailyOsc,
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
            if params.window is None or params.window == 1
            else px.rolling_mean(
                params.window, min_samples=2
            )
        )
        px_slow = (
            px.rolling_mean(
                len(df), min_samples=2
            )
            if params.slow is None
            else px.rolling_mean(
                params.slow, min_samples=2
            )
        )
        # TODO: or default to rms?
        rx_vol = (
            rx.rolling_std(
                len(df), min_samples=2
            )
            if params.vol is None
            else px.rolling_std(
                params.vol, min_samples=2
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
class BarDailyOscDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyOsc.new(ord, fast=4, slow=12, vol=8)
    >>> unit = BarDailyOsc.new(ord, fast=4, slow=12, vol=8, activation="tanh")
    >>> domain = (
    ...     BarDailyOscDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .with_obj(unit)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-osc-4-12-8' [f64]
    [
        0.979022
        null
        null
        1.240631
        1.349691
        1.105384
        0.828809
        0.060248
    ]
    >>> unit.value_key_val(domain).get(unit.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-osc-4-12-8-tanh' [f64]
    [
        0.752642
        null
        null
        0.845635
        0.87398
        0.802425
        0.679836
        0.060175
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyOscDomain"] = "BarDailyOscDomain"

        SUFFIX: ClassVar[str] = "osc"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyOsc
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyOsc.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyOsc",
    "HandleNewBarDailyOsc",
    "BarDailyOscDomain",
]