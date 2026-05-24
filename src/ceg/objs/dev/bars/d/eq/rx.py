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

from ..ident import *

from .etf import EqDailyEtf
# from .stock import EqDailyStock

#  ------------------


@dataclass(frozen=True)
class EqDailyRx(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    """
    >>> ord = EqDailyEtf.new(
    ...     symbol="SPY",
    ...     field="close",
    ...     adjust="full",
    ...     source="frd"
    ... )
    >>> obj = EqDailyRx.new(ord)
    >>> Graph.with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-eq-d-SPY-close-full-frd-rx' [f64]
    [
        9.375071
        6.945601
        null
        5.529652
        null
        null
        -4.650274
        1.877995
    ]
    """

    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    # TODO: or EqDailyStock once we support union

    parent: EqDailyEtf | str
    params: Params

    def self(self) -> EqDailyRx:
        return self
    
    def get_parent(self) -> EqDailyEtf:
        return get_parent(self)

    def encode_type(self, params: EqDailyRx.Params) -> str:
        return "rx"

    def encode_params(self, params: EqDailyRx.Params) -> str:
        return ""

    def handle(self, event: Event, snap: Snap):
        snap = assert_is(snap, EqDailyRx.Snap)
        if isinstance(event, ValueAppend):
            raise NotImplementedError(self)
        event = assert_is(event, ValueUpdate)

        eq = self.get_parent()
        df = eq.get_df()

        # split: adjust by split ratio
        # dividends: 1 - (Dividend Amount / Prior Day's Close Price)

        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if eq.params.adjust in ["pct", "split"]
            else pl.col(VALUE).forward_fill().diff()
            if eq.params.adjust
            in ["raw", "price", "full"]
            # TODO: split only, raw
            else None
        )
        assert e is not None, eq.params.adjust

        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

        _ = self.set_value_from_df(
            df,
            VALUE,
        )
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(df),
        )

eq_daily_rx = new(EqDailyRx, EqDailyRx.Params)

#  ------------------

__all__ = [
    "EqDailyRx",
    "eq_daily_rx",
]
