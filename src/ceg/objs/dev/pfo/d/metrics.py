
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

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

from ...calc import *

from ..root import *
from .bt import *

#  ------------------

@dataclass(frozen=True)
class DailyBtValue():
    bt: int
    metric: Object | str # TODO: rename value?
    index: int = -1

@dataclass(frozen=True)
class DailyBtMetric(
    UseRootConfigs,
):
    parent: UniverseDaily | str
    
    name: str
    values: tuple[DailyBtValue, ...]
    # calc however you see fit, eg. dv01
    # NOTE: optional: eg. if name=dd, will use returns instead

    version: int = 0
    type: Literal["DailyBtMetric"] = "DailyBtMetric"

    def get_parent(self, domain: Domain) -> UniverseDaily:
        return domain.get(self.parent, t=UniverseDaily)
    
    @classmethod
    def new(
        cls,
        parent: UniverseDaily,
        name: str,
        values: tuple[DailyBtValue, ...] = (),
    ):
        return cls(
            parent=parent,
            path=parent.extend_path(name),
            name=name,
            values=values,
        )

@dataclass(frozen=True)
class DailyBtAggregate(
    DailyChunked,
    DependsSelfNew,
    DependsDomainSync,
):
    parent: DailyBtMetric | str
    
    index: int # index of agg in pfolio

    version: int = 0
    type: Literal["DailyBtAggregate"] = "DailyBtAggregate"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyBtAggregateSnap"] = "DailyBtAggregateSnap"

    def get_parent(self, domain: Domain) -> DailyBtMetric:
        return domain.get(self.parent, t=DailyBtMetric)

    @classmethod
    def new(
        cls,
        parent: DailyBtMetric,
        index: int,
    ):
        index_pad = str(index).zfill(3)
        return cls(
            parent=parent,
            path=parent.extend_path(index_pad),
            index=index,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )
    
    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        parent = self.get_parent(domain)
        res: list[Handler] = []

        object_new = isinstance(event, ObjectNew)
        object_change = isinstance(event, ObjectChange)

        if object_new and parent.name == "rx":
            assert self.path == event.obj, (self, event)
            res.append(HandleNewDailyBtSumRx())

        # TODO: if upstream target change

        elif object_change:
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res
    
@dataclass(frozen=True)
class HandleNewDailyBtAggRx(
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
        obj: DailyBtAggregate,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None
        
        parent = obj.get_parent(domain)

        assert parent.name == "rx", parent
        assert len(parent.values)== 0, parent

        univ = parent.get_parent(domain)

        agg = univ.agg[obj.index]

        if len(agg.bt) == 0:
            bts = [
                DailyBt.new(univ, bt)
                for bt in range(len(univ.bt))
            ]
        
        bts = [
            DailyBt.new(univ, bt)
            for bt in agg.bt
        ]
        rxs = [
            DailyBtRx.new(bt) for bt in bts
        ]

        rx_dfs = [
            daily_to_df(rx, domain)
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
class HandleNewDailyBtSumRx(
    HandleNewDailyBtAggRx
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
    "DailyBtValue",
    "DailyBtMetric",
    "DailyBtAggregate",
]