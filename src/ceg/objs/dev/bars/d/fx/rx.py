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

from .px import FxDailyPx

#  ------------------


@dataclass(frozen=True)
class FxDailyRx(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    
    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: FxDailyPx | str
    params: Params

    def self(self) -> FxDailyRx:
        return self
    
    def get_parent(self) -> FxDailyPx:
        return get_parent(self)

    def encode_type(self, params: FxDailyRx.Params) -> str:
        return "rx"

    def encode_params(self, params: FxDailyRx.Params) -> str:
        return ""

    def handle(self, event: Event, snap: Snap):
        snap = assert_is(snap, FxDailyRx.Snap)
        if isinstance(event, ValueAppend):
            raise NotImplementedError(self)
        
        event = assert_is(event, ValueUpdate)

        fx = self.get_parent()

        symbol = fx.params.symbol

        if "USD" not in symbol:
            raise ValueError(self)

        # usdjpy = 153
        # jpyusd = 0.0065
        # if symbol[:3] == "USD":

        # NOTE: assume always flipped at px to symbol[:3]
        n_fgn = pl.col(VALUE)
        # else:
        #     n_fgn = (
        #         1 / pl.col(VALUE)
        #     ).fill_nan(None)

        # n_dom_tmw = (
        #     n_fgn / n_fgn.shift().forward_fill()
        # ) - 1

        e = n_fgn.forward_fill().pct_change()

        df = fx.get_df().with_columns(
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

fx_daily_rx = new(FxDailyRx, FxDailyRx.Params)

#  ------------------


__all__ = [
    "FxDailyRx",
    "fx_daily_rx",
]
