from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
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

from ..ident import *

from ......apis.eq import *

#  ------------------

# SP500 trackers:
# US: SPY, VOO, and IVV
# UK/EU: VUSA, CSPX, and SPXP.

# ETFs:
# XLY (Consumer Discretionary), XLE (Energy), XLU (Utilities), XLB (Materials).
# VONV (Russell 1000 Value ETF), VTWO (Russell 2000 ETF).
# VONG (Russell 1000 Growth ETF).
# SCHD (U.S. Dividend Equity ETF), SCHM (U.S. Mid-Cap ETF), SCHV (U.S. Large-Cap Value ETF), FNDX (Fundamental U.S. Large Company ETF), FNDB (Fundamental U.S. Broad Market ETF), SCHX (U.S. Large-Cap ETF), SCHB (U.S. Broad Market ETF).
# SCHG (U.S. Large-Cap Growth ETF).
# FNDA (Fundamental U.S. Small Company ETF), SCHK (1000 Index® ETF), SCHP (U.S. TIPS ETF), SCYB (High Yield Bond ETF), SCMB (Municipal Bond ETF), SCHA (U.S. Small-Cap ETF), SCHR (Intermediate-Term U.S. Treasury ETF), SCHJ (1-5 Year Corporate Bond ETF), SCHO (Short-Term U.S. Treasury ETF), SCHZ (U.S. Aggregate Bond ETF), SCHI (5-10 Year Corporate Bond ETF), SCHF (International Equity ETF).
# GLL (UltraShort Gold), EEV (UltraShort Emerging Markets), YXI (UltraShort Japan), BZQ (UltraShort Basic Materials).
# Vanguard High Dividend Yield ETF (VYM)


def load_df_frd(
    symbol: str, field: str, adjust: str, alias: str
):
    try:
        return apis.data.frd.read_file(
            os.environ.get("FRD_DATA", "TODO"),
            folder=apis.data.frd.Folders.ETF,
            symbol=symbol,
            suffix=adjust,
            snap="full",  # full
            freq="1day",  # 1day
            #
            start=None,
            end=None,
            update=True,
        ).select("date", pl.col(field).alias(alias))
    except:
        raise ValueError((symbol, adjust, alias))

def load_join_dfs(
    symbol: str,
    field: str,
    f_df: Callable[[str, str, str, str], pl.DataFrame],
    adjust_aliases: list[tuple[str, str]],
):
    dfs = [
        f_df(symbol, field, adjust, alias)
        for adjust, alias in adjust_aliases
    ]
    return apis.pl.join_dfs(
        dfs,
        on="date",
        how="full",
        coalesce=True,
        maintain_order="left_right",
    )

@dataclass(frozen=True)
class EqDailyEtf(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    """
    >>> universe = [
    ...     ("SPY", "pct"),
    ...     ("VONG", "splits"),
    ...     ("VONG", "price"),
    ...     ("VYM", "divs"),
    ...     ("VYM", "full"),
    ... ]
    >>> objs = [
    ...     EqDailyEtf.new(
    ...         eng,
    ...         symbol=symbol,
    ...         field="close",
    ...         adjust=adjust,
    ...         source="frd",
    ...     )
    ...     for symbol, adjust in universe
    ... ]
    >>> (
    ...     spy, splits, price, divs, full
    ... ) = objs
    >>> Graph().with_objs(*objs).sync()
    >>> f_v = lambda o: o.value_key_val().get(o.id)
    >>> f_v(spy).tail(8)
    shape: (8,)
    Series: 'test-eq-d-SPY-close-pct-frd' [f64]
    [
        563.98
        null
        null
        574.08
        575.46
        568.59
        567.08
        555.66
    ]
    >>> v_splits = f_v(splits)
    >>> v_splits.ser.filter(v_splits.ser.is_not_null())
    shape: (1,)
    Series: 'test-eq-d-VONG-close-splits-frd' [f64]
    [
        4.0
    ]
    >>> v_divs = f_v(divs)
    >>> v_divs.ser.filter(v_divs.ser.is_not_null()).tail(8)
    shape: (8,)
    Series: 'test-eq-d-VYM-close-divs-frd' [f64]
    [
        0.008103
        0.007286
        0.0099
        0.00557
        0.008493
        0.006633
        0.007696
        0.006522
    ]
    >>> f_v(price).tail(8)
    shape: (8,)
    Series: 'test-eq-d-VONG-close-price-frd' [f64]
    [
        484.8
        488.44
        null
        490.64
        null
        null
        489.0
        490.8
    ]
    >>> f_v(full).tail(8)
    shape: (8,)
    Series: 'test-eq-d-VYM-close-full-frd' [f64]
    [
        251.829977
        254.082387
        null
        255.930067
        null
        null
        253.378509
        252.586646
    ]
    """

    @dataclass(frozen=True)
    class Params(Daily.Params):
        symbol: str
        field: str
        adjust: str
        source: str
        fx: str | None = None  # ccy to convert to

    parent: str | Date
    params: Params

    def self(self) -> EqDailyEtf:
        return self
    
    def get_parent(self) -> Date:
        return get_parent(self)

    def encode_type(self, params: EqDailyEtf.Params):
        return "eq-d-etf"

    def encode_params(self, params: EqDailyEtf.Params):
        return "-".join(map(str, [
            params.symbol,
            params.field,
            params.adjust,
            params.source,
            # TODO: and fx if given
        ]))

    def handle(self, event: Event, snap: Snap):

        # TODO: possibly later on, deal with obj storage def changes?
        # TODO: and data appends versus full regen

        event = assert_is(event, ValueUpdate)
        snap = assert_is(snap, Daily.Snap)

        tags = self.params
        

        if tags.source == "frd":
            f_df = load_df_frd
        else:
            raise ValueError(self)

        if tags.adjust in ["pct", "split", "raw"]:
            df = f_df(
                tags.symbol,
                tags.field,
                tags.adjust,
                alias=tags.field,
            )
        elif tags.adjust == "splits":
            df = load_join_dfs(
                tags.symbol,
                tags.field,
                f_df,
                [("raw", "raw"), ("split", "split")],
            )
            df = df.pipe(
                with_splits,
                raw="raw",
                split="split",
                alias=tags.field,
            )
        elif tags.adjust == "divs":
            df = load_join_dfs(
                tags.symbol,
                tags.field,
                f_df,
                [
                    ("raw", "raw"),
                    ("split", "split"),
                    ("pct", "pct"),
                ],
            )
            df = df.pipe(
                with_divs,
                raw="raw",
                split="split",
                pct="pct",
                alias=tags.field,
            )
        elif tags.adjust == "price":
            df = load_join_dfs(
                tags.symbol,
                tags.field,
                f_df,
                [
                    ("raw", "raw"),
                    ("split", "split"),
                    ("pct", "pct"),
                ],
            )
            df = df.pipe(
                with_price,
                raw="raw",
                split="split",
                pct="pct",
                alias=tags.field,
            )
        elif tags.adjust == "full":
            df = load_join_dfs(
                tags.symbol,
                tags.field,
                f_df,
                [
                    ("raw", "raw"),
                    ("split", "split"),
                    ("pct", "pct"),
                ],
            )
            df = df.pipe(
                with_full,
                raw="raw",
                split="split",
                pct="pct",
                alias=tags.field,
            )
        else:
            raise ValueError(tags)

        _ = self.set_value_from_df(
            df,
            tags.field,
        )
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(df),
        )
    
eq_daily_etf = new(EqDailyEtf, EqDailyEtf.Params)

#  ------------------

__all__ = [
    "EqDailyEtf",
    "eq_daily_etf",
]