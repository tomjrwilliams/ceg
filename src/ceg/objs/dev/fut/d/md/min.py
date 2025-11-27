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
class FutDailyDMin(FutDailyMD):
    version: int = 0
    type: Literal["FutDailyDMin"] = "FutDailyDMin"

    SUFFIX: ClassVar[str] = "dmin"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyDMinSnap"] = "FutDailyDMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyDMin:
        return HandleNewFutDailyDMin()

@dataclass(frozen=True)
class HandleNewFutDailyDMin(HandleNewDailyDMin):
    pass

@dataclass(frozen=True)
class FutDailyDMinDomain(FutDailyDomain):
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
    >>> obj = FutDailyDMin.new(ord, window=4)
    >>> domain = (
    ...     FutDailyDMinDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmin-4' [f64]
    [
        49.0
        null
        null
        97.25
        11.0
        0.0
        0.0
        0.0
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyDMinDomain"] = "FutDailyDMinDomain"


        SUFFIX: ClassVar[str] = "dmin"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyDMin
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyDMin.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name