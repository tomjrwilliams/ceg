
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

from ....root import *
from ...bars import *
from ...calc.emd import FutDailyEMD, FutDailyEMDDomain, HandleNewFutDailyEMD

@dataclass(frozen=True)
class FutDailyEMDA(FutDailyEMD):
    """FutDailyEMDA(UseRootConfigs):\n
    parent: FutDaily\n
    symbol: str\n
    field: str\n
    adjust: str\n
    fx: str | None = None # ccy to convert to\n
    source: str | None = None\n
    """
    parent: FutDailyOrdinal | FutDailyRx | str
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
    type: Literal["FutDailyEMDA"] = "FutDailyEMDA"

    SUFFIX: ClassVar[str] = "emda"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMDASnap"] = "FutDailyEMDASnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEMDA:
        return HandleNewFutDailyEMDA()

@dataclass(frozen=True)
class HandleNewFutDailyEMDA(HandleNewFutDailyEMD):

    def calc(
        self,
        event: Event,
        obj: FutDailyEMDA,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        
        e = pl.col(obj.field) - (
            pl.col(obj.field).ewm_mean(
                span=obj.span,
                alpha=obj.alpha,
                min_samples=1, # TODO: or more?
            )
        )

        df = df.with_columns(
            pl.when(pl.col(obj.field).is_null())
            .then(None)
            .otherwise(e)
            .alias(obj.field)
        )
        return df
    

@dataclass(frozen=True)
class FutDailyEMDADomain(FutDailyEMDDomain):
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
    >>> obj = FutDailyEMDA.new(ord, span=4)
    >>> domain = (
    ...     FutDailyEMDADomain.new(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emda-4' [f64]
    [
        4.910321
        null
        null
        34.609705
        21.697869
        -24.176808
        -24.998186
        -81.700068
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyEMDDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEMDADomain"] = "FutDailyEMDADomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "emda"
        # TODO: to override

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyEMDA
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEMDA.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name