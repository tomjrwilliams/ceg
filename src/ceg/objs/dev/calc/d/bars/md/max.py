

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
class BarDailyDMax(BarDailyM):
    version: int = 0
    type: Literal["BarDailyDMax"] = "BarDailyDMax"

    SUFFIX: ClassVar[str] = "dmax"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyDMaxSnap"] = "BarDailyDMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyDMax:
        return HandleNewBarDailyDMax()

@dataclass(frozen=True)
class HandleNewBarDailyDMax(HandleNewDailyDMax):
    pass

@dataclass(frozen=True)
class BarDailyDMaxDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyDMax.new(ord, window=4)
    >>> domain = (
    ...     BarDailyDMaxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmax-4' [f64]
    [
        -11.5
        null
        null
        0.0
        0.0
        -67.0
        -87.25
        -203.5
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyDMaxDomain"] = "BarDailyDMaxDomain"

        SUFFIX: ClassVar[str] = "dmax"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyDMax
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyDMax.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name