
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
class BarDailyRms(BarDailyM):
    version: int = 0
    type: Literal["BarDailyRms"] = "BarDailyRms"

    SUFFIX: ClassVar[str] = "rms"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyRmsSnap"] = "BarDailyRmsSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyRms:
        return HandleNewBarDailyRms()

@dataclass(frozen=True)
class HandleNewBarDailyRms(HandleNewDailyRms):
    pass


@dataclass(frozen=True)
class BarDailyRmsDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyRms.new(ord, window=4)
    >>> domain = (
    ...     BarDailyRmsDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-rms-4' [f64]
    [
        0.007817
        null
        null
        0.012045
        0.0121
        0.011903
        0.010457
        0.011816
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyRmsDomain"] = "BarDailyRmsDomain"

        SUFFIX: ClassVar[str] = "rms"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyRms
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyRms.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name