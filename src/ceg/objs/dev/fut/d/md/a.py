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

from ...root import *
from ..bars import *
from ..calc import *

from ....calc import *

@dataclass(frozen=True)
class FutDailyDMA(FutDailyMD):
    version: int = 0
    type: Literal["FutDailyDMA"] = "FutDailyDMA"

    SUFFIX: ClassVar[str] = "dma"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyDMASnap"] = "FutDailyDMASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyDMA:
        return HandleNewFutDailyDMA()

@dataclass(frozen=True)
class HandleNewFutDailyDMA(HandleNewDailyDMA):
    pass

@dataclass(frozen=True)
class FutDailyDMADomain(FutDailyDomain):
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
    >>> obj = FutDailyDMA.new(ord, window=4)
    >>> domain = (
    ...     FutDailyDMADomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dma-4' [f64]
    [
        10.75
        null
        null
        48.625
        5.5
        -41.0
        -45.9375
        -114.0625
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyDMADomain"] = "FutDailyDMADomain"


        SUFFIX: ClassVar[str] = "dma"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyDMA
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyDMA.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name