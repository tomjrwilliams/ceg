
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
class HandleNewFutDailyEMPLast(HandleNewFutDailyEM):

    def calc(
        self,
        event: Event,
        obj: FutDailyEMPLast,
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
            alpha=obj.calc_alpha(),
            l=100,
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
    alpha: float,
    l: int,
    norm: bool,
):
    n: int = len(v)
    acc: float = 0
    res = np.zeros(n, dtype=v.dtype)
    ws = np.zeros(l, dtype=v.dtype)

    j = 0

    alpha_l = alpha ** l
    while j < l:
        ws[j] = (
            (1 - alpha) / (1 - alpha_l)
        ) * (alpha ** j)
        j += 1

    w_sum = np.sum(ws)

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
                    w = ws[k]
                    acc += vv * w
                    w_acc += w
                    k += 1
                j += 1
            
            if w_acc > 0:
                scale = w_sum / w_acc
                acc *= scale

                res[i] = acc

                if acc < min_v:
                    min_v = acc
                if acc > max_v:
                    max_v = acc

        i += 1

    return res
    

@dataclass(frozen=True)
class FutDailyEMPLastDomain(FutDailyEMDomain):
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
    >>> obj = FutDailyEMPLast.new(ord, span=4)
    >>> domain = (
    ...     FutDailyEMPLastDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-emplast-4' [f64]
    [
        0.278662
        null
        null
        0.00008
        0.000032
        0.840017
        0.936007
        0.999937
    ]
    """

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
            | FutDailyRx
            | FutDailyEMPLast
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEMPLast.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name