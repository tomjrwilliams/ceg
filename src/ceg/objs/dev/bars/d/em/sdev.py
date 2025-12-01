
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

from ..prelude import *
from ..calc.em import *
from ....calc.d import *

@dataclass(frozen=True)
class BarDailyESDev(BarDailyEM):

    version: int = 0
    type: Literal["BarDailyESDev"] = "BarDailyESDev"

    SUFFIX: ClassVar[str] = "esdev"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyESDevSnap"] = "BarDailyESDevSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyESDev:
        return HandleNewBarDailyESDev()

@dataclass(frozen=True)
class HandleNewBarDailyESDev(HandleNewDailyESDev):
    pass


@dataclass(frozen=True)
class BarDailyESDevDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyESDev.new(ord, span=4)
    >>> domain = (
    ...     BarDailyESDevDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-esdev-4' [f64]
    [
        0.008131
        null
        null
        0.011117
        0.009137
        0.011772
        0.009188
        0.012394
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyESDevDomain"] = "BarDailyESDevDomain"

        SUFFIX: ClassVar[str] = "esdev"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyESDev
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyESDev.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyESDev",
    "HandleNewBarDailyESDev",
    "BarDailyESDevDomain",
]