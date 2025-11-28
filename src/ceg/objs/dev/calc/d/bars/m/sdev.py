
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

from ..prelude import *
from ..base.m import *
from ...base.m import *
from ...handlers.m import *

@dataclass(frozen=True)
class BarDailySDev(BarDailyM):
    version: int = 0
    type: Literal["BarDailySDev"] = "BarDailySDev"

    SUFFIX: ClassVar[str] = "sdev"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailySDevSnap"] = "BarDailySDevSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailySDev:
        return HandleNewBarDailySDev()

@dataclass(frozen=True)
class HandleNewBarDailySDev(HandleNewDailySDev):
    pass


@dataclass(frozen=True)
class BarDailySDevDomain(DailyChunkedDomain):
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
    >>> obj = BarDailySDev.new(ord, window=4)
    >>> domain = (
    ...     BarDailySDevDomain.empty(eng)
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
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailySDevDomain"] = "BarDailySDevDomain"

        SUFFIX: ClassVar[str] = "sdev"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailySDev
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailySDev.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailySDev",
    "HandleNewBarDailySDev",
    "BarDailySDevDomain",
]