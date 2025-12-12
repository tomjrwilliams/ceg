
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
class BarDailyEZDomain(DailyDomain):
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
    >>> obj = DailyEZ.new(ord, span=4)
    >>> domain = (
    ...     BarDailyEZDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-ez-4' [f64]
    [
        0.211555
        null
        null
        0.978895
        0.682668
        -0.267117
        -0.460222
        -0.969193
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyEZDomain"] = "BarDailyEZDomain"

        SUFFIX: ClassVar[str] = "ez"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | DailyEZ
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | DailyEZ.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyEZDomain",
]