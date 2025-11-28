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
from ..base.emd import *
# from ...handlers.em import *
from ...handlers.emd import *

@dataclass(frozen=True)
class BarDailyDEMin(BarDailyDEM):
    version: int = 0
    type: Literal["BarDailyDEMin"] = "BarDailyDEMin"

    SUFFIX: ClassVar[str] = "demin"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyDEMinSnap"] = "BarDailyDEMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyDEMin:
        return HandleNewBarDailyDEMin()

@dataclass(frozen=True)
class HandleNewBarDailyDEMin(HandleNewDailyDEMin):
    pass

@dataclass(frozen=True)
class BarDailyDEMinDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyDEMin.new(ord, span=4)
    >>> domain = (
    ...     BarDailyDEMinDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-demin-4' [f64]
    [
        10.248
        null
        null
        64.4988
        45.29928
        0.0
        0.0
        0.0
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyDEMinDomain"] = "BarDailyDEMinDomain"

        SUFFIX: ClassVar[str] = "demin"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyDEMin
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyDEMin.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "BarDailyDEMin",
    "HandleNewBarDailyDEMin",
    "BarDailyDEMinDomain",
]