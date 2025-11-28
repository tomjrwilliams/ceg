
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

@dataclass(frozen=True)
class BarDailyEMA(BarDailyEM):

    version: int = 0
    type: Literal["BarDailyEMA"] = "BarDailyEMA"

    SUFFIX: ClassVar[str] = "ema"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyEMASnap"] = "BarDailyEMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyEMA:
        return HandleNewBarDailyEMA()

@dataclass(frozen=True)
class HandleNewBarDailyEMA(HandleNewDailyEMA):
    pass

@dataclass(frozen=True)
class BarDailyEMADomain(DailyChunkedDomain):
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
    >>> obj = BarDailyEMA.new(ord, span=4)
    >>> domain = (
    ...     BarDailyEMADomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ema-4' [f64]
    [
        5711.550199
        null
        null
        5753.13012
        5782.478072
        5773.286843
        5759.672106
        5705.003264
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyEMADomain"] = "BarDailyEMADomain"
        SUFFIX: ClassVar[str] = "ema"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyEMA
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyEMA.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "BarDailyEMA",
    "HandleNewBarDailyEMA",
    "BarDailyEMADomain",
]