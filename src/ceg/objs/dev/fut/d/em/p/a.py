# backward freq of above em a
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

from ....root import Fut, FutDaily, ORD, CONTR
from ...bars.ord import FutDailyOrdinal
from ...calc.emp import FutDailyEMP, FutDailyEMPDomain

@dataclass(frozen=True)
class FutDailyEMPA(FutDailyEMP):
    """FutDailyEMPA(UseRootConfigs):\n
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
    type: Literal["FutDailyEMPA"] = "FutDailyEMPA"

    SUFFIX: ClassVar[str] = "empa"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMPASnap"] = "FutDailyEMPASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEMPA:
        return HandleNewFutDailyEMPA()

@dataclass(frozen=True)
class HandleNewFutDailyEMPA(Handler):

    def calc(
        self,
        event: Event,
        obj: FutDailyEMPA,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        
        e = (
            pl.col(obj.field).ewm_mean(
                span=obj.span,
                alpha=obj.alpha,
                min_samples=2, # TODO: or more?
            )
        )
        # now backward rolling window
        # average over indicator series, gt ^
        raise ValueError((obj, "TODO"))
    

@dataclass(frozen=True)
class FutDailyEMPADomain(Domain):

    @dataclass(frozen=True)
    class Root(
        FutDailyEMPDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEMPADomain"] = "FutDailyEMPADomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "empa"
        # TODO: to override

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyEMPA
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyEMPA.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name