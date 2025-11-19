
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

from ....root import *
from ...bars import *
from ...calc import *

@dataclass(frozen=True)
class FutDailyMDMin(FutDailyMD):
    """FutDailyMDMin(UseRootConfigs):\n
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

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyMDMin"] = "FutDailyMDMin"

    SUFFIX: ClassVar[str] = "mdmin"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyMDMinSnap"] = "FutDailyMDMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyMDMin:
        return HandleNewFutDailyMDMin()

@dataclass(frozen=True)
class HandleNewFutDailyMDMin(HandleNewFutDailyMD):

    def calc(
        self,
        event: Event,
        obj: FutDailyMDMin,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        
        e = pl.col(obj.field) - (
            pl.col(obj.field).rolling_min(
                len(df) if obj.window is None else obj.window,
                min_samples=1, # TODO: or more?
            )
        )

        df = df.with_columns(
            pl.when(pl.col(obj.field).is_null())
            .then(None)
            .otherwise(e)
            .alias(obj.field)
        )
        return df
    

@dataclass(frozen=True)
class FutDailyMDMinDomain(FutDailyMDDomain):
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
    >>> obj = FutDailyMDMin.new(ord, window=4)
    >>> domain = (
    ...     FutDailyMDMinDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-mdmin-4' [f64]
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

    @dataclass(frozen=True)
    class Root(
        FutDailyMDDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyMDMinDomain"] = "FutDailyMDMinDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "mdmin"
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
            | FutDailyMDMin
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyMDMin.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name