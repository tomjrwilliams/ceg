
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
class FutDailyMed(FutDailyM):
    version: int = 0
    type: Literal["FutDailyMed"] = "FutDailyMed"

    SUFFIX: ClassVar[str] = "med"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyMedSnap"] = "FutDailyMedSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyMed:
        return HandleNewFutDailyMed()

@dataclass(frozen=True)
class HandleNewFutDailyMed(HandleNewDailyMed):
    pass

@dataclass(frozen=True)
class FutDailyMedDomain(FutDailyDomain):
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
    >>> obj = FutDailyMed.new(ord, window=4)
    >>> domain = (
    ...     FutDailyMedDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-med-4' [f64]
    [
        5715.5
        null
        null
        5766.875
        5821.0
        5815.5
        5787.5
        5749.375
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyMedDomain"] = "FutDailyMedDomain"


        SUFFIX: ClassVar[str] = "med"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyMed
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyMed.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name