
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
from ..calc import *

@dataclass(frozen=True)
class FutDailyEMPLast(FutDailyEM):
    """FutDailyEMPLast(UseRootConfigs):\n
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

    span: float | None = None
    alpha: float | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyEMPLast"] = "FutDailyEMPLast"

    SUFFIX: ClassVar[str] = "emplast"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMPLastSnap"] = "FutDailyEMPLastSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEMPLast:
        return HandleNewFutDailyEMPLast()

@dataclass(frozen=True)
class HandleNewFutDailyEMPLast(Handler):

    def calc(
        self,
        event: Event,
        obj: FutDailyEMPLast,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        # rolling window, indicator series of gt last
        # then exp decay over binary series?
        # so actually needs sliding window of weights
        raise ValueError((obj, "TODO"))
    

@dataclass(frozen=True)
class FutDailyEMPLastDomain(Domain):

    @dataclass(frozen=True)
    class Root(
        FutDailyEMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEMPLastDomain"] = "FutDailyEMPLastDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "emplast"
        # TODO: to override

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyEMPLast
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyEMPLast.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name