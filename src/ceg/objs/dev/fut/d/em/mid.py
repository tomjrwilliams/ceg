
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
class FutDailyEMid(FutDailyEM):
    """FutDailyEMid(UseRootConfigs):\n
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
    type: Literal["FutDailyEMid"] = "FutDailyEMid"

    SUFFIX: ClassVar[str] = "emid"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMidSnap"] = "FutDailyEMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEMid:
        return HandleNewFutDailyEMid()

@dataclass(frozen=True)
class HandleNewFutDailyEMid(Handler):

    def calc(
        self,
        event: Event,
        obj: FutDailyEMid,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        raise ValueError((obj, "TODO"))
    

@dataclass(frozen=True)
class FutDailyEMidDomain(Domain):

    @dataclass(frozen=True)
    class Root(
        FutDailyEMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEMidDomain"] = "FutDailyEMidDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "emid"
        # TODO: to override

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyEMid
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyEMid.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name