
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
class FutDailyPLast(FutDailyM):
    """FutDailyPLast(UseRootConfigs):\n
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
class HandleNewFutDailyPLast(HandleNewFutDailyM):

    def calc(
        self,
        event: Event,
        obj: FutDailyPLast,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        s = df.get_column(obj.field)
        v = s.fill_null(0).to_numpy()
        is_null = s.is_null().to_numpy()
        res = calc(
            v,
            v,
            is_null,
            l=len(df) if obj.window is None else obj.window,
            norm = False, # TODO: parametrise in path?
        )

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

# rolling window, indicator series of gt last
# then exp decay over binary series?
# so actually needs sliding window of weights

@nb.jit()
def calc(
    v: np.ndarray,
    loc: np.ndarray,
    is_null: np.ndarray,
    l: int,
    norm: bool,
):
    n: int = len(v)
    acc: float = 0
    res = np.zeros(n, dtype=v.dtype)

    l -= 1

    j = 0

    min_v = 1
    max_v = 0

    i: int = 0
    while i < n:
        if i == 0:
            acc = 0.5
            res[i] = acc
        elif not is_null[i]:
            j = 1
            k = 0

            acc = 0
            w_acc = 0

            while k < l and j <= i:
                ij = i - j
                if not is_null[ij]:
                    vv = (
                        0 
                        if v[ij] < loc[i]
                        else 0.5 if v[ij] == loc[i]
                        else 1
                    )
                    w = 1 / (l + 1)
                    acc += vv * w
                    w_acc += w
                    k += 1
                j += 1
            
            if w_acc > 0:
                scale = 1 / w_acc
                acc *= scale

                res[i] = acc

                if acc < min_v:
                    min_v = acc
                if acc > max_v:
                    max_v = acc

        i += 1
        
    return res
    

@dataclass(frozen=True)
class FutDailyPLastDomain(FutDailyMDomain):
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
    >>> obj = FutDailyPLast.new(ord, window=4)
    >>> domain = (
    ...     FutDailyPLastDomain.new(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-plast-4' [f64]
    [
        0.333333
        null
        null
        0.0
        0.0
        0.666667
        1.0
        1.0
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
            | FutDailyRx
            | FutDailyPLast
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyPLast.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name