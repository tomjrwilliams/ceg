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

#  ------------------

@dataclass(frozen=True)
class FxDailyPx(
    Daily,
    DependsParent,
    HasDataFrameValue
):

    @dataclass(frozen=True)
    class Params(Daily.Params):
        symbol: str
        field: str
        adjust: str
        source: str
        fx: str | None = None  # ccy to convert to

    parent: str | Date
    params: Params

    def self(self) -> FxDailyPx:
        return self
    
    def get_parent(self) -> Date:
        return get_parent(self)

    def encode_type(self, params: FxDailyPx.Params):
        return "fx-d"

    def encode_params(self, params: FxDailyPx.Params):
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
            df = apis.data.frd.read_file(
                os.environ.get("FRD_DATA", "TODO"),
                folder=apis.data.frd.Folders.FX,
                symbol=tags.symbol,
                suffix=tags.adjust,
                snap="full",  # full
                freq="1day",  # 1day
                #
                start=None,
                end=None,
                update=True,
            ).select("date", tags.field)
        else:
            raise ValueError(self)

        if "USD" == tags.symbol[3:]:
            pass
        else:
            df = df.with_columns(
                (1 / pl.col(tags.field)).alias(tags.field)
            )

        _ = self.set_value_from_df(
            df,
            tags.field,
        )
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(df),
        )

fx_daily_px = new(FxDailyPx, FxDailyPx.Params)

#  ------------------

__all__ = [
    "FxDailyPx",
    "fx_daily_px",
]
