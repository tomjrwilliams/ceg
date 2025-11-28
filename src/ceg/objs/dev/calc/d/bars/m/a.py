
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

from ..prelude import *

@dataclass(frozen=True)
class BarDailyMA(BarDailyM):
    version: int = 0
    type: Literal["BarDailyMA"] = "BarDailyMA"

    SUFFIX: ClassVar[str] = "ma"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyMASnap"] = "BarDailyMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyMA:
        return HandleNewBarDailyMA()

@dataclass(frozen=True)
class HandleNewBarDailyMA(HandleNewDailyMA):
    pass

@dataclass(frozen=True)
class BarDailyMADomain(DailyChunkedDomain):
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
    >>> obj = BarDailyMA.new(ord, window=4)
    >>> domain = (
    ...     BarDailyMADomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ma-4' [f64]
    [
        5707.5
        null
        null
        5766.875
        5821.0
        5800.5
        5785.1875
        5737.0625
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyMADomain"] = "BarDailyMADomain"

        SUFFIX: ClassVar[str] = "ma"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyMA
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyMA.Snap
        )