
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

from .... import apis
from ....apis import t
from ....apis import kv
from ....engine import *

from ..calc import *

from .collections import *

#  ------------------

@dataclass(frozen=True)
class DailyAggregate(
    DailyChunked,
    DependsSelfNew,
    DependsParentValue,
):
    parent: Collection | str
    
    metric: str
    index: int # index of agg in pfolio

    version: int = 0
    type: Literal["DailyAggregate"] = "DailyAggregate"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyAggregateSnap"] = "DailyAggregateSnap"

    def get_parent(self, domain: Domain) -> Collection:
        return domain.get(self.parent, t=Collection)

    @classmethod
    def new(
        cls,
        parent: Collection,
        metric: str,
        index: int,
    ):
        index_pad = str(index).zfill(3)
        return cls(
            parent=parent,
            path=parent.extend_path(index_pad),
            metric=metric,
            index=index,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        # TODO: rx of the bts in agg (or all if empty)
        # no parent?
        # then inherit from tracks latest upstream
        return (
            self.depends_self(domain)
            + self.depends_parent(domain)
        )
    
    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        parent = self.get_parent(domain)
        res: list[Handler] = []

        obj = domain.get(event.obj, Object)

        object_new = isinstance(event, ObjectNew)
        object_change = isinstance(event, ObjectChange)

        parent_update = (
            isinstance(event, ValueUpdate)
            and obj.path == parent.path
        )

        if parent_update and self.metric == "hsum":
            res.append(HandleNewDailyHSum())

        # TODO: if upstream target change

        elif object_change:
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res
    
@dataclass(frozen=True)
class HandleNewDailyHAgg(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):

    def agg_rx(
        self,
        rxs: pl.DataFrame,
        first: dt.date,
        last: dt.date,
    ) -> pl.DataFrame:
        ...

    def handle(
        self,
        event: ObjectNew,
        obj: DailyAggregate,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None
        
        parent = univ = obj.get_parent(domain)

        agg = univ.agg[obj.index]

        if len(agg.objs) == 0:
            rxs = [
                univ.get_object(
                    i, domain, Daily
                )
                for i in range(
                    len(univ.objects)
                )           
            ]
        else:
            rxs = [
                univ.get_object(
                    i, domain, Daily
                )
                for i in agg.objs
            ]

        rx_dfs = [
            daily_pl_df(rx, domain)
            for rx in rxs
        ]
        rx_df = apis.pl.join_dfs(
            *rx_dfs,
            on="date",
            how="full",
            coalesce=True,
            maintain_order="left_right",
        ).sort(by="date")

        d0 = rx_df.select(pl.col("date").min()).item()
        d1 = rx_df.select(pl.col("date").max()).item()

        df = self.agg_rx(rx_df, d0, d1)

        assert len(df) == (
            (d1 - d0).days + 1
        ), (df, d1, d0)

        e, v = self.set_value_from_df(
            obj,
            domain,
            df,
            VALUE,
            publish=True,
            t = self.key_value_type(obj, domain),
        )

        snap = self.set_snap_from_df(
            df, event, obj, domain
        )

        return e


@dataclass(frozen=True)
class HandleNewDailyHSum(
    HandleNewDailyHAgg
):

    def agg_rx(
        self,
        rxs: pl.DataFrame,
        first: dt.date,
        last: dt.date,
    ) -> pl.DataFrame:
        return rxs.select(
            pl.col("date"),
            pl.sum_horizontal(pl.exclude("date"))
            .alias(VALUE)
        )

__all__ = [
    "DailyAggregate",
]