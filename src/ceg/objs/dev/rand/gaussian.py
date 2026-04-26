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
    TypeAlias,
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

DAILY_GAUSSIAN = "DailyGaussian"

class Interface(BaseInterface):

    @dataclass(frozen=True)
    class Args(BaseInterface.Args):
        parent: Project | str | None
        start: dt.date
        end: dt.date
        mu: float = 0.
        sd: float = 1.
        walk: bool = False
        suffix: str = ""
        # NOTE: suffix for dismabiguation


    @dataclass(frozen=True)
    class Params(Args):
        uid: str | None = None
        xid: int | str | None = None

    @dataclass(frozen=True)
    class Object(
        BaseInterface.Object,
        Args,
        UseRootDataFrameStorage,
        DependsGraphSync, 
        Daily,
    ):

        @dataclass(frozen=True)
        class Snap(Daily.Snap):
            type: Literal["DailyGaussian"] = DAILY_GAUSSIAN
        
        def handler(self, event: Event, snap: Snap) -> Handler:
            if isinstance(event, GraphSync):
                return Interface.Handler()
            # elif isinstance(event, ValueUpdate):
            raise ValueError((self, event))
        
    @dataclass(frozen=True)
    class Handler(
        HandleNewValueFromPl,
        HandleNewDailyFromDf,
    ):
        def handle(
            self,
            event: ObjectNew | GraphSync,
            snap: DailyGaussian.Snap,
            obj: DailyGaussian,
        ):
            n = (obj.end - obj.start).days
            v = np.random.normal(obj.mu, obj.sd, (n+1,))
            if obj.walk:
                v = np.cumsum(v)

            df = pl.DataFrame({
                obj.id: v
            }).with_columns(pl.date_range(obj.start, obj.end).alias("date"))

            v = self.set_value_from_df(
                obj,
                df,
                obj.id,
                t=self.key_value_type(obj),
            )

            new_snap = snap.with_dates(first=obj.start, last=obj.end)

            # NOTE: if we only issue event on snap change
            # we update the graph sync values, but dont re issue event

            # if (
            #     new_snap.first != snap.first
            #     and new_snap.last != snap.last
            # ):
            return ValueUpdate.from_snap(
                self.update_snap_from_df(new_snap, df),
                into=frozendict(),
            )

@dataclass(frozen=True)
class DailyGaussian(Interface.Object):
    
    version: int = 0
    type: Literal["DailyGaussian"] = DAILY_GAUSSIAN

new = init_new(Interface.Params, DailyGaussian)
bind = init_bind(new, DailyGaussian)

Type: TypeAlias = DailyGaussian

__all__ = ["Interface", "Type", "new", "bind"]