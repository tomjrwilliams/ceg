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

from .....engine import *

from .h import trans_prob

NAME = "TransitionProbSignal"

class Interface(BaseInterface):

    @dataclass(frozen=True)
    class Args(BaseInterface.Args):
        parent: Daily | str

        n_bwd: int
        n_fwd: int

        bwd: int | None
        fwd: int

        # NOTE: assume given == bwd, if bwd has value

    @dataclass(frozen=True)
    class Params(Args):
        uid: str | None = None
        xid: int | str | None = None

    @dataclass(frozen=True)
    class Object(
        BaseInterface.Object,
        Args,
        UseRootDataFrameStorage,
        DependsParentValue,
        Daily
    ):

        @dataclass(frozen=True)
        class Snap(Daily.Snap):
            type: Literal["TransitionProbSignal"] = NAME
        
        def handler(self, event: Event, snap: Snap) -> Handler:
            if isinstance(event, ValueUpdate):
                return Interface.Handler()
            # elif isinstance(event, ValueUpdate):
            raise ValueError((self, event))
        
    @dataclass(frozen=True)
    class Handler(
        HandleNewDailyFromDf,
        HandleNewValueFromParentPl,
        HandleNewDailyFromDailyParent,
    ):
        def handle(
            self,
            event: ValueUpdate,
            snap: Object.Snap,
            obj: Object,
        ):
            # TODO: assert is transiton prob?
            parent = context.get().get(obj.parent, Daily)

            df_probs = daily_pl_df_cached(parent, alias = parent.id)

            e = pl.col(parent.id)

            df = df_probs.select(
                "date",
                (
                    pl.lit(True)
                    if obj.bwd is None
                    else trans_prob.get_state_indicator(
                        e,
                        obj.n_bwd,
                        obj.n_fwd,
                        bwd=obj.bwd,
                    ).cast(pl.Boolean)
                ).alias("active").forward_fill(),
                trans_prob.get_transition_prob(
                    e,
                    obj.n_bwd,
                    obj.n_fwd,
                    bwd=obj.bwd,
                    fwd=obj.fwd,
                    given="bwd" if obj.bwd is not None else None
                ).alias("prob").forward_fill()
            )

            try:
                df = df.select(
                    "date",
                    pl.when(pl.col("active"))
                    .then("prob")
                    .when(pl.col("active").shift(1))
                    .then(0.)
                    .otherwise(None)
                    .alias(obj.id)
                )
            except Exception as e:
                raise ValueError(df, list(df_probs.iter_rows())[:2]) from e

            v = self.set_value_from_df(
                obj,
                df,
                obj.id,
                t=self.key_value_type(obj),
            )

            new_snap = self.update_snap_from_parent(
                snap, obj, parent=parent,
            )

            if (
                new_snap.first != snap.first
                and new_snap.last != snap.last
            ):

                return ValueUpdate.from_snap(
                    self.update_snap_from_df(new_snap, df),
                    into=frozendict(),
                )

            return frozendict().set(obj.id, (new_snap, []))

@dataclass(frozen=True)
class TransitionProbSignal(Interface.Object):
    
    version: int = 0
    type: Literal["TransitionProbSignal"] = NAME

new = init_new(Interface.Params, TransitionProbSignal)

Type: TypeAlias = TransitionProbSignal

__all__ = ["Interface", "Type", "new"]