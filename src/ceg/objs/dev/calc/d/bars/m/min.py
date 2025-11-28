
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

from ..base import *
from ...base import *

@dataclass(frozen=True)
class BarDailyMin(BarDailyM):
    version: int = 0
    type: Literal["BarDailyMin"] = "BarDailyMin"

    SUFFIX: ClassVar[str] = "min"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyMinSnap"] = "BarDailyMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyMin:
        return HandleNewBarDailyMin()

@dataclass(frozen=True)
class HandleNewBarDailyMin(HandleNewDailyMin):
    pass


@dataclass(frozen=True)
class BarDailyMinDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyMin.new(ord, window=4)
    >>> domain = (
    ...     BarDailyMinDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-min-4' [f64]
    [
        5669.25
        null
        null
        5718.25
        5815.5
        5759.5
        5739.25
        5623.0
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyMinDomain"] = "BarDailyMinDomain"

        SUFFIX: ClassVar[str] = "min"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyMin
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyMin.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name