
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
class BarDailyERng(BarDailyEM):

    version: int = 0
    type: Literal["BarDailyERng"] = "BarDailyERng"

    SUFFIX: ClassVar[str] = "erng"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyERngSnap"] = "BarDailyERngSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyERng:
        return HandleNewBarDailyERng()
    
@dataclass(frozen=True)
class HandleNewBarDailyERng(HandleNewDailyERng):
    pass

@dataclass(frozen=True)
class BarDailyERngDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyERng.new(ord, span=4)
    >>> domain = (
    ...     BarDailyERngDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-erng-4' [f64]
    [
        13.068
        null
        null
        64.4988
        45.29928
        40.2
        36.27
        91.512
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyERngDomain"] = "BarDailyERngDomain"

        SUFFIX: ClassVar[str] = "erng"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyERng
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyERng.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name