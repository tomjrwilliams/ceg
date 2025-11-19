
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
class FutDailyEMin(FutDailyEM):
    """FutDailyEMin(UseRootConfigs):\n
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
    type: Literal["FutDailyEMin"] = "FutDailyEMin"

    SUFFIX: ClassVar[str] = "emin"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMinSnap"] = "FutDailyEMinSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEMin:
        return HandleNewFutDailyEMin()

@dataclass(frozen=True)
class HandleNewFutDailyEMin(HandleNewFutDailyEM):

    def calc(
        self,
        event: Event,
        obj: FutDailyEMin,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        s = df.get_column(obj.field)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc(v, is_null, alpha=obj.calc_alpha())
        return pl.DataFrame({
            "mask": s,
            obj.field: res
        }).select(
            pl.when(pl.col("mask").is_null())
            .then(None)
            .otherwise(obj.field)
            .alias(obj.field)
        )

import numba as nb

@nb.jit()
def calc(
    v: np.ndarray,
    is_null: np.ndarray,
    alpha: float,
    # TODO: assert float?
):
    i: int = 0
    n: int = len(v)
    res = np.empty(n, dtype=v.dtype)
    acc: float = 0
    alpha_comp = 1 - alpha
    while i < n:
        if not is_null[i]:
            vv = v[i]
            if i == 0:
                acc = vv
            elif vv < acc:
                acc = vv
            else:
                acc = (alpha_comp * acc) + (alpha * vv)
            res[i] = acc
        i += 1
    return res
    

@dataclass(frozen=True)
class FutDailyEMinDomain(FutDailyEMDomain):
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
    >>> obj = FutDailyEMin.new(ord, span=4)
    >>> domain = (
    ...     FutDailyEMinDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emin-4' [f64]
    [
        5708.002
        null
        null
        5751.0012
        5781.20072
        5759.5
        5739.25
        5623.0
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyEMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEMinDomain"] = "FutDailyEMinDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "emin"
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
            | FutDailyEMin
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEMin.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name