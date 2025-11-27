
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
class FutDailyMin(FutDailyM):
    version: int = 0
    type: Literal["FutDailyMin"] = "FutDailyMin"

    SUFFIX: ClassVar[str] = "min"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyMinSnap"] = "FutDailyMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyMin:
        return HandleNewFutDailyMin()

@dataclass(frozen=True)
class HandleNewFutDailyMin(HandleNewDailyMin):
    pass


@dataclass(frozen=True)
class FutDailyMinDomain(FutDailyDomain):
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
    >>> obj = FutDailyMin.new(ord, window=4)
    >>> domain = (
    ...     FutDailyMinDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-min-4' [f64]
    [
        5669.25
        null
        null
        5718.25
        5815.5
        5759.5
        5739.25
        5623.0
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyMinDomain"] = "FutDailyMinDomain"


        SUFFIX: ClassVar[str] = "min"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyMin
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyMin.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name