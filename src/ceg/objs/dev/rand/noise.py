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
class DailyNoise(UseRootDataFrameStorage, DependsParentValue, Daily):

    parent: DailyGaussian | str | None
    mu: float = 0.
    sd: float = 1.
    suffix: str = ""

    version: int = 0
    type: Literal["DailyNoise"] = "DailyNoise"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyNoise"] = "DailyNoise"

    @classmethod
    def new(
        cls, 
        parent: Project,
        mu: float = 0., 
        sd: float = 1.,
        suffix: str = ""
    ):
        # TODO: suffix into id?
        return cls(
            id=cls.build_id(parent, "noise", mu, sd,suffix),
            parent=parent,
            suffix=suffix,
            mu=mu,
            sd=sd,
        )


    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            return HandleNewDailyNoise()
        # elif isinstance(event, ValueUpdate):
        raise ValueError((self, event))


@dataclass(frozen=True)
class HandleNewDailyNoise(
    HandleNewDailyFromDailyParentPl,
    HandleNewDailyFromDf,
):
    def handle(
        self,
        event: ObjectNew | GraphSync,
        snap: DailyNoise.Snap,
        obj: DailyNoise,
    ):
        parent = context.get().get(obj.parent, DailyGaussian)
        df = daily_pl_df(
            parent,
            alias="v"
        )
        v = np.random.normal(loc=obj.mu, scale=obj.sd, size=(len(df),))
        # s = pl.Series("noise", v)

        try:
            df = df.select(
                "date",
                pl.col("v").add(v).alias(obj.id)
            )
        except Exception as e:
            raise ValueError(df) from e

        v = self.set_value_from_df(
            obj,
            df,
            obj.id,
            t=self.key_value_type(obj),
        )
        new_snap = self.update_snap_from_parent(
            snap, obj, parent=parent,
        )

        # if (
        #     new_snap.first != snap.first
        #     and new_snap.last != snap.last
        # ):
        return ValueUpdate.from_snap(
            self.update_snap_from_df(new_snap, df),
            into=frozendict(),
        )

        # return frozendict().set(obj.id, (new_snap, []))
    

__all__ = ["DailyNoise"]