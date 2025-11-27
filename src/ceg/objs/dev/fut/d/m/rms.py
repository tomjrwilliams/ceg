
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
class FutDailyRms(FutDailyM):
    version: int = 0
    type: Literal["FutDailyRms"] = "FutDailyRms"

    SUFFIX: ClassVar[str] = "rms"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyRmsSnap"] = "FutDailyRmsSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyRms:
        return HandleNewFutDailyRms()

@dataclass(frozen=True)
class HandleNewFutDailyRms(HandleNewDailyRms):
    pass


@dataclass(frozen=True)
class FutDailyRmsDomain(FutDailyDomain):
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
    >>> obj = FutDailyRms.new(ord, window=4)
    >>> domain = (
    ...     FutDailyRmsDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-rms-4' [f64]
    [
        0.007817
        null
        null
        0.012045
        0.0121
        0.011903
        0.010457
        0.011816
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyRmsDomain"] = "FutDailyRmsDomain"


        SUFFIX: ClassVar[str] = "rms"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyRms
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyRms.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name