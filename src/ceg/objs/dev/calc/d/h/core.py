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

from ......engine import *
from ....agg import *

from .rank import *
from .quantize import *
from .trans_prob import *

NAME = "DailyHAggregation"

class Interface(Aggregation.Interface):

    @dataclass(frozen=True)
    class Args(Aggregation.Interface.Args):
        
        func: str
        
        kwargs: t.frozen.dict[str, bool | int | float | str] = frozendict()

    @dataclass(frozen=True)
    class Params(Args):
        uid: str | None = None
        xid: int | str | None = None

    @dataclass(frozen=True)
    class Object(
        Aggregation.Interface.Object,
        Args,
        UseRootDataFrameStorage,
        Daily,
    ):
        
        @dataclass(frozen=True)
        class Snap(Aggregation.Type.Snap, Daily.Snap):
            pass

        def handler(self, event: Event, snap: Snap) -> Handler:
            if isinstance(event, ValueUpdate):
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
            event: ValueUpdate,
            snap: Object.Snap,
            obj: Object,
        ):
            g = context.get()
            params = [
                g.get(p, Daily) for p in obj.params
            ]
            
            df = daily_pl_df(params)

            df = AGGREGATES[obj.func](
                df,
                obj.id,
                params=obj.params,
                kwargs = obj.kwargs,
            )
            print("HANDLE AGG")

            v = self.set_value_from_df(
                obj,
                df,
                obj.id,
                t=self.key_value_type(obj),
            )

            new_snap = self.update_snap_from_df(
                snap,
                df,
            )

            # NOTE: tbc if we should always update the snap?
            # even if the date range doesnt change?
            # presumably - the upstram can filter for if not to fire?
            
            if obj.kwargs.get("fire_every_time"):
                return ValueUpdate.from_snap(
                    self.update_snap_from_df(new_snap, df),
                    into=frozendict(),
                )

            elif (
                new_snap.first != snap.first
                and new_snap.last != snap.last
            ):
                return ValueUpdate.from_snap(
                    self.update_snap_from_df(new_snap, df),
                    into=frozendict(),
                )
            
            return frozendict().set(obj.id, (new_snap, []))

def aggregate_sum(
    df: pl.DataFrame,
    alias: str,
    params: tuple[str, ...] | frozendict[str, str],
    kwargs: frozendict[str, bool | int | float | str],
):
    if isinstance(params, frozendict):
        params = tuple(params.values())
    return df.select(
        "date",
        pl.sum_horizontal(*[
            pl.col(k) for k in params
        ]).alias(alias)
    )

AGGREGATES = {
    "sum": aggregate_sum,
    "rank": aggregate_rank,
    "quantize": aggregate_quantize,
    "transition_prob": aggregate_transition_probs,
}

@dataclass(frozen=True)
class DailyHSum(Interface.Object):
    func: str = "sum"
    kwargs: t.frozen.dict[str, bool | int | float | str] = frozendict()
    
    version: int = 0
    type: Literal["DailyHSum"] = "DailyHSum"

    @dataclass(frozen=True)
    class Snap(Interface.Object.Snap):
        type: Literal["DailyHSum"] = "DailyHSum"

sum = init_new(Interface.Params, DailyHSum)
bind_sum = init_bind(sum, DailyHSum)
Sum: TypeAlias = DailyHSum

@dataclass(frozen=True)
class DailyHProd(Interface.Object):
    func: str = "prod"
    kwargs: t.frozen.dict[str, bool | int | float | str] = frozendict()
    
    version: int = 0
    type: Literal["DailyHProd"] = "DailyHProd"

    @dataclass(frozen=True)
    class Snap(Interface.Object.Snap):
        type: Literal["DailyHProd"] = "DailyHProd"

prod = init_new(Interface.Params, DailyHProd)
bind_prod = init_bind(prod, DailyHProd)
Prod: TypeAlias = DailyHProd

@dataclass(frozen=True)
class DailyHRank(Interface.Object):
    func: str = "rank"
    kwargs: t.frozen.dict[str, bool | int | float | str] = frozendict()
    
    version: int = 0
    type: Literal["DailyHRank"] = "DailyHRank"

    @dataclass(frozen=True)
    class Snap(Interface.Object.Snap):
        type: Literal["DailyHRank"] = "DailyHRank"

rank = init_new(Interface.Params, DailyHRank)
bind_rank = init_bind(rank, DailyHRank)
Rank: TypeAlias = DailyHRank

@dataclass(frozen=True)
class DailyHQuantize(Interface.Object):
    func: str = "quantize"
    kwargs: t.frozen.dict[str, bool | int | float | str] = frozendict()
    
    version: int = 0
    type: Literal["DailyHQuantize"] = "DailyHQuantize"

    @dataclass(frozen=True)
    class Snap(Interface.Object.Snap):
        type: Literal["DailyHQuantize"] = "DailyHQuantize"

quantize = init_new(Interface.Params, DailyHQuantize)
bind_quantize = init_bind(quantize, DailyHQuantize)
Quantize: TypeAlias = DailyHQuantize

@dataclass(frozen=True)
class DailyHTransitionProb(Interface.Object):
    func: str = "trans_prob"
    kwargs: t.frozen.dict[str, bool | int | float | str] = frozendict()
    
    version: int = 0
    type: Literal["DailyHTransitionProb"] = "DailyHTransitionProb"

    @dataclass(frozen=True)
    class Snap(Interface.Object.Snap):
        type: Literal["DailyHTransitionProb"] = "DailyHTransitionProb"

transition_prob = init_new(Interface.Params, DailyHTransitionProb)
bind_transition_prob = init_bind(transition_prob, DailyHTransitionProb)
TransitionProb: TypeAlias = DailyHTransitionProb

__all__ = [
    "Interface",
    "Sum",
    "sum",
    "bind_sum",
    "Prod",
    "prod",
    "bind_prod",
    "Rank",
    "rank",
    "bind_rank",
    "TransitionProb",
    "bind_transition_prob",
    "transition_prob"
]