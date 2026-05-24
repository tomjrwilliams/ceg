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

from .gaussian import Type as DailyGaussian

@dataclass(frozen=True)
class DailyNoise(
    Daily,
    DependsParent,
    HasDataFrameValue
):
    @dataclass(frozen=True)
    class Params(Daily.Params):
        mu: float = 0.
        sd: float = 1.

    parent: str | Daily
    params: Params

    def self(self) -> DailyNoise:
        return self
    
    def get_parent(self) -> Daily:
        return get_parent(self)

    def encode_type(self, params: DailyNoise.Params):
        return "noise"

    def encode_params(self, params: DailyNoise.Params):
        return "-".join(map(str, [
            params.mu,
            params.sd
        ]))

    def handle(self, event: Event, snap: Snap):

        event = assert_is(event, ValueUpdate)
        snap = assert_is(snap, DailyNoise.Snap)

        parent = self.get_parent()
        df = parent.get_df(alias=VALUE)

        v = np.random.normal(
            loc=self.params.mu, scale=self.params.sd, size=(len(df),)
        )
        # s = pl.Series("noise", v)

        try:
            df = df.select(
                "date",
                pl.col(VALUE).add(v).alias(self.id)
            )
        except Exception as e:
            raise ValueError(df) from e

        _ = self.set_value_from_df(
            df,
            VALUE,
        )
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(df),
        )

daily_noise = new(DailyNoise, DailyNoise.Params)

__all__ = ["DailyNoise"]