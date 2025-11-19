
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
class FutDailyEMax(FutDailyEM):
    """FutDailyEMax(UseRootConfigs):\n
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
    type: Literal["FutDailyEMax"] = "FutDailyEMax"

    SUFFIX: ClassVar[str] = "emax"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMaxSnap"] = "FutDailyEMaxSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEMax:
        return HandleNewFutDailyEMax()

@dataclass(frozen=True)
class HandleNewFutDailyEMax(HandleNewFutDailyEM):

    def calc(
        self,
        event: Event,
        obj: FutDailyEMax,
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
    mu: float = 0
    alpha_comp = 1 - alpha
    while i < n:
        if not is_null[i]:
            vv = v[i]
            if i == 0:
                acc = vv
                mu = acc
            elif vv > acc:
                acc = vv
            else:
                acc = (alpha_comp * acc) + (alpha * vv)
            res[i] = acc
        i += 1
    return res


@dataclass(frozen=True)
class FutDailyEMaxDomain(FutDailyEMDomain):
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
    >>> obj = FutDailyEMax.new(ord, span=4)
    >>> domain = (
    ...     FutDailyEMaxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emax-4' [f64]
    [
        5721.07
        null
        null
        5815.5
        5826.5
        5799.7
        5775.52
        5714.512
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyEMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEMaxDomain"] = "FutDailyEMaxDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "emax"
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
            | FutDailyEMax
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEMax.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name