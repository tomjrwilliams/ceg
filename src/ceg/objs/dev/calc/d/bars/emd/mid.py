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
class BarDailyDEMid(BarDailyDEM):
    version: int = 0
    type: Literal["BarDailyDEMid"] = "BarDailyDEMid"

    SUFFIX: ClassVar[str] = "demid"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyDEMidSnap"] = "BarDailyDEMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyDEMid:
        return HandleNewBarDailyDEMid()

@dataclass(frozen=True)
class HandleNewBarDailyDEMid(HandleNewDailyDEMid):
    pass
    

@dataclass(frozen=True)
class BarDailyDEMidDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyDEMid.new(ord, span=4)
    >>> domain = (
    ...     BarDailyDEMidDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-demid-4' [f64]
    [
        3.714
        null
        null
        32.2494
        22.64964
        -20.1
        -18.135
        -45.756
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyDEMidDomain"] = "BarDailyDEMidDomain"

        SUFFIX: ClassVar[str] = "demid"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyDEMid
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyDEMid.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name