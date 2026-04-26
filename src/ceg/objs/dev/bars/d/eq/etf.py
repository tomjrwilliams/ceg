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


@dataclass(frozen=True)
class EqDailyEtfTags(BarDailyTags):
    symbol: str
    ident: None
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to


@dataclass(frozen=True)
class EqDailyEtf(
    DependsGraphSync,
    BarDaily,
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

    parent: Project | str | None

    tags: EqDailyEtfTags

    version: int = 0
    type: Literal["EqDailyEtf"] = "EqDailyEtf"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["EqDailyEtfSnap"] = "EqDailyEtfSnap"

    @classmethod
    def new(
        cls,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        parent: Project | str | None = None,
    ):
        (
            ident,
            symbol,
            etf,
            field,
            adjust,
            source,
            fx,
        ) = cls.identifier(
            symbol,
            ident,
            field,
            adjust,
            source,
            fx=fx,
            expect=ETF,
        )
        assert adjust in EQ_ADJUSTMENTS, adjust
        return cls(
            id=cls.build_id(
                parent,
                "eq-etf-d",
                ident,
                strict=False,
            ),
            parent=parent,
            tags=EqDailyEtfTags(
                symbol=symbol,
                ident=etf,
                field=field,
                adjust=adjust,
                fx=fx,
                source=source,
            ),
        )

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, GraphSync):
            return HandleNewEqDailyEtf()
        # elif isinstance(event, ValueUpdate):
        raise ValueError((self, event))


@dataclass(frozen=True)
class HandleUpdateEqDailyEtf(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: EqDailyEtf,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return


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
class HandleNewEqDailyEtf(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
    def handle(
        self,
        event: ObjectChange,
        snap: EqDailyEtf.Snap,
        obj: EqDailyEtf,
    ):
        tags = obj.tags

        if tags.source == "frd":
            f_df = load_df_frd
        else:
            raise ValueError(obj)

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

        v = self.set_value_from_df(
            obj,
            df,
            tags.field,
            t=self.key_value_type(obj),
        )

        new_snap = self.update_snap_from_df(snap, df)
        if (
            new_snap.first != snap.first
            and new_snap.last != snap.last
        ):
            # TODO: new snap!!!??
            return ValueUpdate.from_snap(
                self.update_snap_from_df(snap, df),
                into=frozendict(),
            )

        return frozendict().set(obj.id, (snap, []))


# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------


@dataclass(frozen=True)
class DependsEqDailyEtfParent(DependsParentValue):
    parent: EqDailyEtf | str

    def get_parent(self) -> EqDailyEtf:
        return context.get().get(self.parent, EqDailyEtf)


#  ------------------

__all__ = [
    "EqDailyEtf",
    "HandleUpdateEqDailyEtf",
    "HandleNewEqDailyEtf",
    "DependsEqDailyEtfParent",
]
# TODO: if __name__ == __main__:
# run domain by name
