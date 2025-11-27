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
class FutDailyDMed(FutDailyMD):
    version: int = 0
    type: Literal["FutDailyDMed"] = "FutDailyDMed"

    SUFFIX: ClassVar[str] = "dmed"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyDMedSnap"] = "FutDailyDMedSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyDMed:
        return HandleNewFutDailyDMed()

@dataclass(frozen=True)
class HandleNewFutDailyDMed(HandleNewDailyDMed):
    pass
    

@dataclass(frozen=True)
class FutDailyDMedDomain(FutDailyDomain):
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
    >>> obj = FutDailyDMed.new(ord, window=4)
    >>> domain = (
    ...     FutDailyDMedDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmed-4' [f64]
    [
        2.75
        null
        null
        48.625
        5.5
        -56.0
        -48.25
        -126.375
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyDMedDomain"] = "FutDailyDMedDomain"


        SUFFIX: ClassVar[str] = "dmed"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyDMed
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyDMed.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name