
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
class BarDailyZWDomain(DailyDomain):
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
    >>> obj = DailyZW.new(ord, window=4)
    >>> domain = (
    ...     BarDailyZWDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-zw-4' [f64]
    [
        0.406191
        null
        null
        0.707107
        0.707107
        -1.141092
        -1.083191
        -1.346812
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyZWDomain"] = "BarDailyZWDomain"

        SUFFIX: ClassVar[str] = "zw"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | DailyZW
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | DailyZW.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyZWDomain",
]