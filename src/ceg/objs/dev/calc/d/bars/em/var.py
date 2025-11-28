
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
from ..base.em import *
from ...base.em import *
from ...handlers.em import *

@dataclass(frozen=True)
class BarDailyEVar(BarDailyEM):

    version: int = 0
    type: Literal["BarDailyEVar"] = "BarDailyEVar"

    SUFFIX: ClassVar[str] = "evar"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyEVarSnap"] = "BarDailyEVarSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyEVar:
        return HandleNewBarDailyEVar()

@dataclass(frozen=True)
class HandleNewBarDailyEVar(HandleNewDailyEVar):
    pass


@dataclass(frozen=True)
class BarDailyEVarDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyEVar.new(ord, span=4)
    >>> domain = (
    ...     BarDailyEVarDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-evar-4' [f64]
    [
        0.000066
        null
        null
        0.000124
        0.000083
        0.000139
        0.000084
        0.000154
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyEVarDomain"] = "BarDailyEVarDomain"

        SUFFIX: ClassVar[str] = "evar"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyEVar
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyEVar.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyEVar",
    "HandleNewBarDailyEVar",
    "BarDailyEVarDomain",
]