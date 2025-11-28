
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
class BarDailySkew(BarDailyM):
    version: int = 0
    type: Literal["BarDailySkew"] = "BarDailySkew"

    SUFFIX: ClassVar[str] = "skew"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailySkewSnap"] = "BarDailySkewSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailySkew:
        return HandleNewBarDailySkew()

@dataclass(frozen=True)
class HandleNewBarDailySkew(HandleNewDailySkew):
    pass


@dataclass(frozen=True)
class BarDailySkewDomain(DailyChunkedDomain):
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
    >>> obj = BarDailySkew.new(ord, window=4)
    >>> domain = (
    ...     BarDailySkewDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-skew-4' [f64]
    [
        0.15138
        null
        null
        2.6070e-14
        3.3046e-14
        0.073938
        0.464041
        -0.228863
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailySkewDomain"] = "BarDailySkewDomain"

        SUFFIX: ClassVar[str] = "skew"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailySkew
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailySkew.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailySkew",
    "HandleNewBarDailySkew",
    "BarDailySkewDomain",
]