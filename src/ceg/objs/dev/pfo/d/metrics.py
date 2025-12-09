
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
class PfolioPositionMetric():
    bt: int
    metric: Object | str
    index: int = -1

@dataclass(frozen=True)
class PfolioDailyMetric(
    UseRootConfigs,
):
    parent: PfolioDaily | str
    
    name: str
    values: tuple[PfolioPositionMetric, ...]
    # calc however you see fit, eg. dv01
    # NOTE: optional: eg. if name=dd, will use returns instead

    version: int = 0
    type: Literal["PfolioDailyMetric"] = "PfolioDailyMetric"

    def get_parent(self, domain: Domain) -> PfolioDaily:
        return domain.get(self.parent, t=PfolioDaily)
    
    @classmethod
    def new(
        cls,
        parent: PfolioDaily,
        name: str,
        values: tuple[PfolioPositionMetric, ...] = (),
    ):
        return cls(
            parent=parent,
            path=parent.extend_path(name),
            name=name,
            values=values,
        )

@dataclass(frozen=True)
class PfolioDailyAggregate(
    UseChunkedValues,
    DependsSelfNew,
    DependsDomainSync,
):
    parent: PfolioDailyMetric | str
    
    index: int # index of agg in pfolio

    version: int = 0
    type: Literal["PfolioDailyAggregate"] = "PfolioDailyAggregate"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["PfolioDailyAggregateSnap"] = "PfolioDailyAggregateSnap"

    def get_parent(self, domain: Domain) -> PfolioDailyMetric:
        return domain.get(self.parent, t=PfolioDailyMetric)

    @classmethod
    def new(
        cls,
        parent: PfolioDailyMetric,
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
            res.append(HandleNewPfolioDailySumRx())

        # TODO: if upstream target change

        elif object_change:
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res
    
@dataclass(frozen=True)
class HandleNewPfolioDailyAggRx(Handler):

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
        obj: PfolioDailyAggregate,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None
        
        parent = obj.get_parent(domain)

        assert parent.name == "rx", parent
        assert len(parent.values)== 0, parent

        pfolio = parent.get_parent(domain)

        agg = pfolio.agg[obj.index]

        if len(agg.bt) == 0:
            bts = [
                PfolioDailyBt.new(pfolio, bt)
                for bt in range(len(pfolio.bt))
            ]
        
        bts = [
            PfolioDailyBt.new(pfolio, bt)
            for bt in agg.bt
        ]
        rxs = [
            PfolioDailyRx.new(bt) for bt in bts
        ]

        rx_snap = rxs[0].get_snap(domain)
        assert rx_snap is not None, rx_snap
        assert isinstance(rx_snap, PfolioDailyRx.Snap), rx_snap

        rx_dfs = [
            chunked_daily_to_df(rx, domain)
            for rx in rxs
        ]
        rx_df = apis.pl.join_dfs(
            *rx_dfs,
            on="date",
            how="full",
            coalesce=True,
            maintain_order="left_right",
        )

        df = self.agg_rx(rx_df, rx_snap.first, rx_snap.last)

        e, _ = set_value_chunk_from_df(
            obj,
            domain,
            df,
            VALUE,
            chunk=kv.BODY,
            publish=True,
        )

        snap = (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                rx_snap.first,
                rx_snap.last,
            ), PfolioDailyAggregate.Snap)
        )

        return e


@dataclass(frozen=True)
class HandleNewPfolioDailySumRx(
    HandleNewPfolioDailyAggRx
):

    def agg_rx(
        self,
        rxs: pl.DataFrame,
        first: dt.date,
        last: dt.date,
    ) -> pl.DataFrame:
        return rxs.drop("date").select(
            pl.sum_horizontal(pl.all()).alias(VALUE)
        )

__all__ = [
    "PfolioPositionMetric",
    "PfolioDailyMetric",
    "PfolioDailyAggregate",
]