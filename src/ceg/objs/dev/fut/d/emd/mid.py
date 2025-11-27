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
class FutDailyDEMid(FutDailyDEM):
    version: int = 0
    type: Literal["FutDailyDEMid"] = "FutDailyDEMid"

    SUFFIX: ClassVar[str] = "demid"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyDEMidSnap"] = "FutDailyDEMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyDEMid:
        return HandleNewFutDailyDEMid()

@dataclass(frozen=True)
class HandleNewFutDailyDEMid(HandleNewDailyDEMid):
    pass
    

@dataclass(frozen=True)
class FutDailyDEMidDomain(FutDailyDomain):
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
    >>> obj = FutDailyDEMid.new(ord, span=4)
    >>> domain = (
    ...     FutDailyDEMidDomain.empty(eng)
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
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyDEMidDomain"] = "FutDailyDEMidDomain"


        SUFFIX: ClassVar[str] = "demid"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyDEMid
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyDEMid.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name