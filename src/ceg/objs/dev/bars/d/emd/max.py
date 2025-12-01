

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

@dataclass(frozen=True)
class BarDailyDEMax(BarDailyDEM):
    version: int = 0
    type: Literal["BarDailyDEMax"] = "BarDailyDEMax"

    SUFFIX: ClassVar[str] = "demax"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyDEMaxSnap"] = "BarDailyDEMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyDEMax:
        return HandleNewBarDailyDEMax()

@dataclass(frozen=True)
class HandleNewBarDailyDEMax(HandleNewDailyDEMax):
    pass

@dataclass(frozen=True)
class BarDailyDEMaxDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyDEMax.new(ord, span=4)
    >>> domain = (
    ...     BarDailyDEMaxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-demax-4' [f64]
    [
        -2.82
        null
        null
        0.0
        0.0
        -40.2
        -36.27
        -91.512
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyDEMaxDomain"] = "BarDailyDEMaxDomain"

        SUFFIX: ClassVar[str] = "demax"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyDEMax
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyDEMax.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyDEMax",
    "HandleNewBarDailyDEMax",
    "BarDailyDEMaxDomain",
]