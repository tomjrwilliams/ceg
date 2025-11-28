
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
class BarDailyEMax(BarDailyEM):

    version: int = 0
    type: Literal["BarDailyEMax"] = "BarDailyEMax"

    SUFFIX: ClassVar[str] = "emax"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyEMaxSnap"] = "BarDailyEMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyEMax:
        return HandleNewBarDailyEMax()

@dataclass(frozen=True)
class HandleNewBarDailyEMax(HandleNewDailyEMax):
    pass

@dataclass(frozen=True)
class BarDailyEMaxDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyEMax.new(ord, span=4)
    >>> domain = (
    ...     BarDailyEMaxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emax-4' [f64]
    [
        5721.07
        null
        null
        5815.5
        5826.5
        5799.7
        5775.52
        5714.512
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyEMaxDomain"] = "BarDailyEMaxDomain"

        SUFFIX: ClassVar[str] = "emax"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyEMax
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyEMax.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyEMax",
    "HandleNewBarDailyEMax",
    "BarDailyEMaxDomain",
]