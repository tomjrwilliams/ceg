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

from .px import IndexDailyPx

#  ------------------


@dataclass(frozen=True)
class IndexDailyRx(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    """
    >>> ord = IndexDailyPx.new(
    ...     symbol="VIX",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd"
    ... )
    >>> obj = IndexDailyRx.new(ord, adjust="abs")
    >>> Graph.with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-idx-d-VIX-close-raw-frd-rx' [f64]
    [
        -0.52
        null
        null
        -1.8
        -0.33
        1.18
        0.36
        2.96
    ]
    """

    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: IndexDailyPx | str
    params: Params

    def self(self) -> IndexDailyRx:
        return self
    
    def get_parent(self) -> IndexDailyPx:
        return get_parent(self)

    def encode_type(self, params: IndexDailyRx.Params) -> str:
        return "rx"

    def encode_params(self, params: IndexDailyRx.Params) -> str:
        return ""

    def handle(self, event: Event, snap: Snap):
        snap = assert_is(snap, IndexDailyRx.Snap)
        if isinstance(event, ValueAppend):
            raise NotImplementedError(self)
        event = assert_is(event, ValueUpdate)

        eq = self.get_parent()
        df = eq.get_df()

        parent = self.get_parent()

        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if parent.params.adjust == "pct"
            else pl.col(VALUE).forward_fill().diff()
            if parent.params.adjust == "abs"
            else None
        )
        assert e is not None, parent.params.adjust

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

index_daily_rx = new(IndexDailyRx, IndexDailyRx.Params)

#  ------------------
__all__ = [
    "IndexDailyRx",
    "index_daily_rx",
]
