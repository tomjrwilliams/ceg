
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

from ....calc.d import *



@dataclass(frozen=True)
class BarDailyEMinDomain(DailyDomain):
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
    >>> obj = DailyEMin.new(ord, span=4)
    >>> domain = (
    ...     BarDailyEMinDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emin-4' [f64]
    [
        5708.002
        null
        null
        5751.0012
        5781.20072
        5759.5
        5739.25
        5623.0
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyEMinDomain"] = "BarDailyEMinDomain"

        SUFFIX: ClassVar[str] = "emin"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | DailyEMin
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | DailyEMin.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyEMinDomain",
]