
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
class BarDailyERms(BarDailyEM):

    version: int = 0
    type: Literal["BarDailyERms"] = "BarDailyERms"

    SUFFIX: ClassVar[str] = "erms"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyERmsSnap"] = "BarDailyERmsSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyERms:
        return HandleNewBarDailyERms()

@dataclass(frozen=True)
class HandleNewBarDailyERms(HandleNewDailyERms):
    pass


@dataclass(frozen=True)
class BarDailyERmsDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyERms.new(ord, span=4)
    >>> domain = (
    ...     BarDailyERmsDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-erms-4' [f64]
    [
        0.007089
        null
        null
        0.012077
        0.009431
        0.010308
        0.008288
        0.014329
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyERmsDomain"] = "BarDailyERmsDomain"

        SUFFIX: ClassVar[str] = "erms"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyERms
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyERms.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name