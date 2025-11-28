
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
from ..base import *
from ...base import *
from ...handlers import *

@dataclass(frozen=True)
class BarDailyMax(BarDailyM):
    version: int = 0
    type: Literal["BarDailyMax"] = "BarDailyMax"

    SUFFIX: ClassVar[str] = "max"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyMaxSnap"] = "BarDailyMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyMax:
        return HandleNewBarDailyMax()

@dataclass(frozen=True)
class HandleNewBarDailyMax(HandleNewDailyMax):
    pass

@dataclass(frozen=True)
class BarDailyMaxDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyMax.new(ord, window=4)
    >>> domain = (
    ...     BarDailyMaxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-max-4' [f64]
    [
        5729.75
        null
        null
        5815.5
        5826.5
        5826.5
        5826.5
        5826.5
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyMaxDomain"] = "BarDailyMaxDomain"

        SUFFIX: ClassVar[str] = "max"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyMax
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyMax.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyMax",
    "HandleNewBarDailyMax",
    "BarDailyMaxDomain",
]