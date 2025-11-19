
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
class FutDailyERng(FutDailyEM):
    """FutDailyERng(UseRootConfigs):\n
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
    type: Literal["FutDailyERng"] = "FutDailyERng"

    SUFFIX: ClassVar[str] = "erng"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyERngSnap"] = "FutDailyERngSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyERng:
        return HandleNewFutDailyERng()

from .max import calc as calc_max
from .min import calc as calc_min

@dataclass(frozen=True)
class HandleNewFutDailyERng(HandleNewFutDailyEM):

    def calc(
        self,
        event: Event,
        obj: FutDailyERng,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        s = df.get_column(obj.field)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        mx = calc_max(v, is_null, alpha=obj.calc_alpha())
        mn = calc_min(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "mask": s,
            "max": mx,
            "min": mn,
        }).select(
            pl.when(pl.col("mask").is_null())
            .then(None)
            .otherwise(
                pl.col("max") - pl.col("min")
            )
            .alias(obj.field)
        )


@dataclass(frozen=True)
class FutDailyERngDomain(FutDailyEMDomain):
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
    >>> obj = FutDailyERng.new(ord, span=4)
    >>> domain = (
    ...     FutDailyERngDomain.empty(eng)
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
        FutDailyEMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyERngDomain"] = "FutDailyERngDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "erng"
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
            | FutDailyERng
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyERng.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name