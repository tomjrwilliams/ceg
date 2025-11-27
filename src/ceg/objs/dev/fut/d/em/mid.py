
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
class FutDailyEMid(FutDailyEM):

    version: int = 0
    type: Literal["FutDailyEMid"] = "FutDailyEMid"

    SUFFIX: ClassVar[str] = "emid"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMidSnap"] = "FutDailyEMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEMid:
        return HandleNewFutDailyEMid()

@dataclass(frozen=True)
class HandleNewFutDailyEMid(HandleNewDailyEMid):
    pass

@dataclass(frozen=True)
class FutDailyEMidDomain(FutDailyDomain):
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
    >>> obj = FutDailyEMid.new(ord, span=4)
    >>> domain = (
    ...     FutDailyEMidDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emid-4' [f64]
    [
        5714.536
        null
        null
        5783.2506
        5803.85036
        5779.6
        5757.385
        5668.756
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEMidDomain"] = "FutDailyEMidDomain"


        SUFFIX: ClassVar[str] = "emid"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyEMid
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEMid.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name