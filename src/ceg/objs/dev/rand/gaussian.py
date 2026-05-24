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
from pathlib import Path
import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

import datetime as dt
from frozendict import frozendict

import numpy as np
import polars as pl

from ....engine import *


@dataclass(frozen=True)
class DailyGaussian(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    # TODO: or daily rand and paramtrise dist into params?
    
    @dataclass(frozen=True)
    class Params(Daily.Params):
        mu: float = 0.
        sd: float = 1.
        cum: bool = False

    parent: str | Date
    params: Params

    def self(self) -> DailyGaussian:
        return self
    
    def get_parent(self) -> Date:
        return get_parent(self)

    def encode_type(self, params: DailyGaussian.Params):
        return "gaussian-d"

    def encode_params(self, params: DailyGaussian.Params):
        return "-".join(map(str, [
            params.mu,
            params.sd
        ]))

    def handle(self, event: Event, snap: Snap):
        
        params = self.params

        event = assert_is(event, ValueUpdate)
        snap = assert_is(snap, DailyGaussian.Snap)

        first, last = (
            assert_is(snap.first, dt.date),
            assert_is(snap.last, dt.date),
        )

        n = (last - first).days
        v = np.random.normal(params.mu, params.sd, (n+1,))

        if params.cum:
            v = np.cumsum(v)

        df = pl.DataFrame({
            self.id: v
        }).with_columns(pl.date_range(first, last).alias("date"))

        _ = self.set_value_from_df(
            df,
            self.id,
        )
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(df),
        )


daily_gaussian = new(
    DailyGaussian,
    DailyGaussian.Params,
)

#  ------------------

__all__ = [
    "DailyGaussian",
    "daily_gaussian"
]