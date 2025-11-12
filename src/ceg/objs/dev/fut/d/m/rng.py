
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

from ...root import *
from ..bars import *
from ..calc import *

@dataclass(frozen=True)
class FutDailyRng(FutDailyM):
    """FutDailyRng(UseRootConfigs):\n
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

    symbol: str
    ident: str
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

    window: int | None = None
    width: float | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyRng"] = "FutDailyRng"

    SUFFIX: ClassVar[str] = "rng"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyRngSnap"] = "FutDailyRngSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyRng:
        return HandleNewFutDailyRng()

@dataclass(frozen=True)
class HandleNewFutDailyRng(HandleNewFutDailyM):

    def calc(
        self,
        event: Event,
        obj: FutDailyRng,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        
        e = pl.col(obj.field)
        u = (
            e.rolling_max(
                len(df) if obj.window is None else obj.window,
                min_samples=1, # TODO: or more?
            )
            if obj.width is None
            else e.rolling_quantile(
                0.5 + obj.width,
                window_size=len(df) if obj.window is None else obj.window,
                min_samples=1, # TODO: or more?
            )
        )
        l = (
            e.rolling_min(
                len(df) if obj.window is None else obj.window,
                min_samples=1, # TODO: or more?
            )
            if obj.width is None
            else e.rolling_quantile(
                0.5 - obj.width,
                window_size=len(df) if obj.window is None else obj.window,
                min_samples=1, # TODO: or more?
            )
        )
        e = u - l

        df = df.with_columns(
            pl.when(pl.col(obj.field).is_null())
            .then(None)
            .otherwise(e)
            .alias(obj.field)
        )
        return df
    

@dataclass(frozen=True)
class FutDailyRngDomain(FutDailyMDomain):
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
    >>> obj = FutDailyRng.new(ord, window=4)
    >>> domain = (
    ...     FutDailyRngDomain.new(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rng-4' [f64]
    [
        60.5
        null
        null
        97.25
        11.0
        67.0
        87.25
        203.5
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyRngDomain"] = "FutDailyRngDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "rng"
        # TODO: to override

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyRng
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyRng.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name