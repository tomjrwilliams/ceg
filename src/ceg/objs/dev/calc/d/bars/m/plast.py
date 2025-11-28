
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
from ..base.m import *
from ...base.m import *
from ...handlers.m import *

@dataclass(frozen=True)
class BarDailyPLast(BarDailyM):
    version: int = 0
    type: Literal["BarDailyPLast"] = "BarDailyPLast"

    SUFFIX: ClassVar[str] = "plast"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyPLastSnap"] = "BarDailyPLastSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyPLast:
        return HandleNewBarDailyPLast()

@dataclass(frozen=True)
class HandleNewBarDailyPLast(HandleNewDailyPLast):
    pass

@dataclass(frozen=True)
class BarDailyPLastDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyPLast.new(ord, window=4)
    >>> domain = (
    ...     BarDailyPLastDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-plast-4' [f64]
    [
        0.333333
        null
        null
        0.0
        0.0
        0.666667
        1.0
        1.0
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyPLastDomain"] = "BarDailyPLastDomain"

        SUFFIX: ClassVar[str] = "plast"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyPLast
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyPLast.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyPLast",
    "HandleNewBarDailyPLast",
    "BarDailyPLastDomain",
]