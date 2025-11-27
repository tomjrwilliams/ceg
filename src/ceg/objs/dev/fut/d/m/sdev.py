
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
class FutDailySDev(FutDailyM):
    version: int = 0
    type: Literal["FutDailySDev"] = "FutDailySDev"

    SUFFIX: ClassVar[str] = "sdev"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailySDevSnap"] = "FutDailySDevSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailySDev:
        return HandleNewFutDailySDev()

@dataclass(frozen=True)
class HandleNewFutDailySDev(HandleNewDailySDev):
    pass


@dataclass(frozen=True)
class FutDailySDevDomain(FutDailyDomain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> ord = FutDailyRx.new(
    ...     eng,
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = FutDailySDev.new(ord, window=4)
    >>> domain = (
    ...     FutDailySDevDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-sdev-4' [f64]
    [
        0.009001
        null
        null
        0.011345
        0.010688
        0.014262
        0.012023
        0.009659
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailySDevDomain"] = "FutDailySDevDomain"


        SUFFIX: ClassVar[str] = "sdev"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailySDev
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailySDev.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name