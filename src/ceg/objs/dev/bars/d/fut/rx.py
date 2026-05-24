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
from .ord import (
    FutDailyOrdinal,
)

#  ------------------


@dataclass(frozen=True)
class FutDailyRx(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    
    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: FutDailyOrdinal | str
    params: Params

    def self(self) -> FutDailyRx:
        return self
    
    def get_parent(self) -> FutDailyOrdinal:
        return get_parent(self)

    def encode_type(self, params: FutDailyRx.Params) -> str:
        return "rx"

    def encode_params(self, params: FutDailyRx.Params) -> str:
        return ""

    def handle(self, event: Event, snap: Snap):
        snap = assert_is(snap, FutDailyRx.Snap)
        if isinstance(event, ValueAppend):
            raise NotImplementedError(self)
        event = assert_is(event, ValueUpdate)
        fut = self.get_parent()
        assert fut.id == event.obj, (
            self,
            event,
        )
        df = fut.get_df()
        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if fut.params.adjust == "pct"
            else pl.col(VALUE).forward_fill().diff()
            if fut.params.adjust == "abs"
            else None
        )
        assert e is not None, fut.params.adjust

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

fut_daily_rx = new(FutDailyRx, FutDailyRx.Params)

#  ------------------


__all__ = [
    "FutDailyRx",
    "fut_daily_rx",
]
