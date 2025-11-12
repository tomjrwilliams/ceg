
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

from ...root import *
from ..bars import *
from ..calc.m import FutDailyM, FutDailyMDomain

@dataclass(frozen=True)
class FutDailyPLast(FutDailyM):
    """FutDailyPLast(UseRootConfigs):\n
    parent: FutDaily\n
    symbol: str\n
    field: str\n
    adjust: str\n
    fx: str | None = None # ccy to convert to\n
    source: str | None = None\n
    """
    parent: FutDailyOrdinal | str
    # TODO: or contract
    # TODO: or rx thereof

    symbol: str
    ident: str
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

    window: int | None = None
    width: float | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyPLast"] = "FutDailyPLast"

    SUFFIX: ClassVar[str] = "plast"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyPLastSnap"] = "FutDailyPLastSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyPLast:
        return HandleNewFutDailyPLast()

@dataclass(frozen=True)
class HandleNewFutDailyPLast(Handler):

    def calc(
        self,
        event: Event,
        obj: FutDailyPLast,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        
        e = pl.col(obj.field)
        # per window, compare each value to last
        # average of greater than last
        raise ValueError((obj, "TODO"))
        return df
    

@dataclass(frozen=True)
class FutDailyPLastDomain(Domain):
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
    >>> obj = FutDailyVar.new(ord, span=4)
    >>> domain = (
    ...     FutDailyVarDomain.new(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-evar-4' [f64]
    [
        0.000056
        null
        null
        null
        0.000024
        0.000081
        0.000038
        0.000106
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyPLastDomain"] = "FutDailyPLastDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "plast"
        # TODO: to override

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyPLast
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyPLast.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name