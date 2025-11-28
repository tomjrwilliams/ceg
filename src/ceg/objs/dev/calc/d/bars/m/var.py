
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
class BarDailyVar(BarDailyM):
    version: int = 0
    type: Literal["BarDailyVar"] = "BarDailyVar"

    SUFFIX: ClassVar[str] = "var"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyVarSnap"] = "BarDailyVarSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyVar:
        return HandleNewBarDailyVar()

@dataclass(frozen=True)
class HandleNewBarDailyVar(HandleNewDailyVar):
    pass


@dataclass(frozen=True)
class BarDailyVarDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyVar.new(ord, window=4)
    >>> domain = (
    ...     BarDailyVarDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-var-4' [f64]
    [
        0.000081
        null
        null
        0.000129
        0.000114
        0.000203
        0.000145
        0.000093
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyVarDomain"] = "BarDailyVarDomain"

        SUFFIX: ClassVar[str] = "var"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyVar
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyVar.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyVar",
    "HandleNewBarDailyVar",
    "BarDailyVarDomain",
]