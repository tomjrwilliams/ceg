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

from ..prelude import *
from ..calc.emd import *
from ....calc.d import *

# TODO: technicalyl given parent dont need the fields themselves?
# the parent string contains all of that implicitly?

@dataclass(frozen=True)
class BarDailyDEMA(BarDailyDEM):
    version: int = 0
    type: Literal["BarDailyDEMA"] = "BarDailyDEMA"

    SUFFIX: ClassVar[str] = "dema"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyDEMASnap"] = "BarDailyDEMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyDEMA:
        return HandleNewBarDailyDEMA()

@dataclass(frozen=True)
class HandleNewBarDailyDEMA(HandleNewDailyDEMA):
    pass

@dataclass(frozen=True)
class BarDailyDEMADomain(DailyChunkedDomain):
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
    >>> obj = BarDailyDEMA.new(ord, span=4)
    >>> domain = (
    ...     BarDailyDEMADomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dema-4' [f64]
    [
        6.699801
        null
        null
        62.36988
        44.021928
        -13.786843
        -20.422106
        -82.003264
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyDEMADomain"] = "BarDailyDEMADomain"

        SUFFIX: ClassVar[str] = "dema"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyDEMA
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyDEMA.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyDEMA",
    "HandleNewBarDailyDEMA",
    "BarDailyDEMADomain",
]