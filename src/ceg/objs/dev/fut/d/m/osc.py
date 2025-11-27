
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
from ......apis.pl import null_mask
from ......engine import *

from ....calc import *
from ...root import *
from ..bars import *
from ..calc import *

@dataclass(frozen=True)
class DailyMOscParams(DailyMParams):
    
    chunk: int = 32
    # TODO: db?

    window: int | None = None
    slow: int | None = None
    vol: int | None = None
    activation: str | None = None

@dataclass(frozen=True)
class FutDailyOsc(FutDailyM):
    """FutDailyOsc(UseRootConfigs):\n
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

    params: DailyMOscParams

    version: int = 0
    type: Literal["FutDailyOsc"] = "FutDailyOsc"

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
        fast: int | None = None,
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
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyOscSnap"] = "FutDailyOscSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyOsc:
        return HandleNewFutDailyOsc()

@dataclass(frozen=True)
class HandleNewFutDailyOsc(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):

    def calc_df(
        self,
        df: pl.DataFrame,
        event: Event,
        obj: FutDailyOsc,
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
class FutDailyOscDomain(FutDailyDomain):
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
    >>> obj = FutDailyOsc.new(ord, fast=4, slow=12, vol=8)
    >>> unit = FutDailyOsc.new(ord, fast=4, slow=12, vol=8, activation="tanh")
    >>> domain = (
    ...     FutDailyOscDomain.empty(eng)
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
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyOscDomain"] = "FutDailyOscDomain"

        SUFFIX: ClassVar[str] = "osc"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyOsc
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyOsc.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name