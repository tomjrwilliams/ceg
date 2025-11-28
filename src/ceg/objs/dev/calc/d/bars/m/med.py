
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
class BarDailyMed(BarDailyM):
    version: int = 0
    type: Literal["BarDailyMed"] = "BarDailyMed"

    SUFFIX: ClassVar[str] = "med"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyMedSnap"] = "BarDailyMedSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyMed:
        return HandleNewBarDailyMed()

@dataclass(frozen=True)
class HandleNewBarDailyMed(HandleNewDailyMed):
    pass

@dataclass(frozen=True)
class BarDailyMedDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyMed.new(ord, window=4)
    >>> domain = (
    ...     BarDailyMedDomain.empty(eng)
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
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyMedDomain"] = "BarDailyMedDomain"

        SUFFIX: ClassVar[str] = "med"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyMed
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyMed.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name