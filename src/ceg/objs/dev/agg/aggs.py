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
import datetime
import datetime as dt

import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass
from functools import partial

from frozendict import frozendict

import numpy as np
import polars as pl
import numba as nb

from .... import apis
from ....apis import assert_is, field_default, t, kv, logs
from ....engine import *

from ..calc import *

from .containers import *

from ....apis.calcs.regimes.dd import *

from ....apis.calcs import *

logger, log = logs.logger(
    __file__, streams=dict(formatter=True)
)

#  ------------------


@dataclass(frozen=True)
class DailyAggregate(
    DailyChunked,
    DependsParentValue,
):
    parent: Container | str

    agg: str
    args: tuple[int, ...]  # args of agg in container
    # NOTE: tuple to allow eg. panel vs panel regression

    params: (
        t.frozen.dict[str, bool | int | float | None] | None
    ) = None

    peers: tuple[str, ...] = ()

    version: int = 0
    type: Literal["DailyAggregate"] = "DailyAggregate"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyAggregateSnap"] = (
            "DailyAggregateSnap"
        )

    def get_parent(self) -> Container:
        return context.get().get(self.parent, t=Container)

    def depends(self) -> list[Dependency]:
        return self.depends_parent() + [
            Dependency(
                self.id,
                peer,
                events=('ValueUpdate',)
            )
            for peer in self.peers
        ]

    @classmethod
    def new(
        cls,
        parent: Container,
        agg: str,
        aggs: (
            int
            | tuple[int, ...]
            | Aggregate
            | tuple[Aggregate, ...]
        ),
        **params: bool | int | float | None,
    ):
        args = (
            (aggs,)
            if isinstance(aggs, (int, Aggregate))
            else aggs
        )
        args = tuple(
            [
                arg if isinstance(arg, int) else arg.index
                for arg in args
            ]
        )
        args_pad = "agg-" + "-".join(map(str, args))
        if len(params):
            # TODO: v much TBC!! need keys?
            # if always a full list, maybe not

            # TODO: None, minus, .0 etc. abbreviate

            args_pad += "-" + "-".join(
                map(str, params.values())
            )
        return cls(
            parent=parent,
            id=parent.extend_id(args_pad),
            agg=agg,
            args=args,
            params=frozendict(params),
        )

    def handler(self, event: Event, snap: Snap) -> Handler:
        parent = self.get_parent()

        # obj = context.get().get(event.obj, Object)

        parent_update = (
            isinstance(event, ValueUpdate)
            and event.obj == parent.id
        )

        if parent_update:
            return agg_handler(self.agg, self.params)

        # TODO: if upstream target change

        raise ValueError((self, event))

    def deriv(
        self,
        key: str | None = None,
        index: int | None = None,
    ):
        return DailyAggregateDeriv.new(
            self, index=index, key=key
        )


def get_agg_dfs(aggs: tuple[int, ...], coll: Container):

    dfs = []

    d0 = d1 = None

    g = context.get(Graph)

    for a in aggs:
        try:
            agg = coll.agg[a]
        except:
            raise ValueError((coll, a))

        objs = agg.get_objects(coll)

        df = (
            apis.pl.join_dfs(
                [daily_pl_df(obj) for obj in objs],
                on="date",
                align=True,
            )
            .sort(by="date")
            .select("date", *[obj.id for obj in objs])
        )

        _d0, _d1 = apis.pl.select_items(
            df,
            (pl.col("date").min(), pl.col("date").max()),
            dt.date,
        )
        d0 = _d0 if d0 is None else min([d0, _d0])
        d1 = _d1 if d1 is None else min([d1, _d1])

        dfs.append(df)

    d0, d1 = assert_is(d0), assert_is(d1)

    return dfs, d0, d1


@dataclass(frozen=True)
class DailyAggregateDeriv(
    DailyChunked,
):
    parent: DailyAggregate | str

    index: int | None = None
    key: str | None = None

    version: int = 0
    type: Literal["DailyAggregateDeriv"] = (
        "DailyAggregateDeriv"
    )

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyAggregateDerivSnap"] = (
            "DailyAggregateDerivSnap"
        )

    def get_parent(self) -> DailyAggregate:
        return context.get().get(
            self.parent, DailyAggregate
        )

    @classmethod
    def new(
        cls,
        parent: DailyAggregate,
        index: int | None = None,
        key: str | None = None,
    ):
        key = key or field_default(cls, "key", str)
        assert index is not None or key is not None, (
            index,
            key,
        )
        id=None if index is None else str(index)
        id=key if id is None else f"{key}-{id}"
        return cls(
            parent=parent,
            id=parent.extend_id(assert_is(id)),
            index=index,
            key=key,
        )

    def depends(self) -> list[Dependency]:
        return []


@dataclass(frozen=True)
class HandleNewDailyAggregate(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        *dfs: pl.DataFrame,
    ) -> (
        pl.DataFrame
        | tuple[
            pl.DataFrame, list[str | int | tuple[str, int]]
        ]
    ): ...

    def handle(
        self,
        event: ObjectChange,
        snap: DailyAggregate.Snap,
        obj: DailyAggregate,
    ):
        # log("INFO")( agg=obj.id)

        g = context.get(Graph)

        container= obj.get_parent()

        dfs, d0, d1 = get_agg_dfs(obj.args, container)

        dfs = dfs + [
            daily_pl_df(g.get(peer, Daily)) for peer in obj.peers
        ]

        if snap.first and snap.last:
            if d0 == snap.first and d1 == snap.last:
                return frozendict().set(
                    obj.id, (snap, ())
                )

        res = self.agg_dfs(d0, d1, *dfs)

        df = None

        # log("INFO")( store=obj.id)

        snaps = frozendict()

        if isinstance(res, pl.DataFrame):
            df = res
            assert len(df) == ((d1 - d0).days + 1), (
                df,
                d1,
                d0,
            )

            v = self.set_value_from_df(
                obj,
                df,
                VALUE,
                t=self.key_value_type(obj),
            )
        elif isinstance(res, tuple):
            df, k_is = res

            assert len(df) == ((d1 - d0).days + 1), (
                df,
                d1,
                d0,
            )

            _snaps = {}

            def store_child(
                k_i: str | int | tuple[str, int],
            ):
                if isinstance(k_i, str):
                    k = col = k_i
                    i = None
                elif isinstance(k_i, int):
                    i = k_i
                    col = f"v_{i}"
                    k = None
                else:
                    k, i = k_i
                    col = f"{k}_{i}"
                child = obj.deriv(
                    key=k,
                    index=i,
                )

                context.get().with_obj(child, force=True)

                child_snap = child.get_snap(Daily.Snap)
                _ = self.set_value_from_df(
                    child,
                    df,
                    col,
                    t=self.key_value_type(child),
                )
                _snaps[child.id] = (
                    replace(
                        child_snap,
                        first=d0,
                        last=d1,
                    ),
                    ValueUpdate(child.id),
                )

            # TODO: fold / accumulate
            _ = list(map(store_child, k_is))

            for k, v in _snaps.items():
                assert k not in snaps, k
                snaps = snaps.set(k, v)
        else:
            raise ValueError((type(res), self))

        # log(agg=obj.id, mess="setting snap")

        return ValueUpdate.from_snap(
            self.update_snap_from_df(
                snap,
                assert_is(df),
                d_min=d0,
                d_max=d1,
            ),
            into=snaps,
        )


@dataclass(frozen=True)
class HandleNewDailyHSum(HandleNewDailyAggregate):
    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        return df.select(
            pl.col("date"),
            h_sum(
                [
                    k
                    for k in df.schema.keys()
                    if k != "date"
                ],
                propagate_nulls=False,
            )
            .alias(VALUE)
            .fill_nan(None),
        )


@dataclass(frozen=True)
class HandleNewDailyHGross(HandleNewDailyAggregate):
    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        df = df.select(
            pl.col("date"),
            h_sum(
                [
                    k
                    for k in df.schema.keys()
                    if k != "date"
                ],
                f=lambda e: e.abs().fill_null(0.0),
                propagate_nulls=False,
            )
            .alias(VALUE)
            .fill_nan(None),
        )
        return df


@dataclass(frozen=True)
class HandleNewDailyHProd(HandleNewDailyAggregate):
    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        df = df.select(
            pl.col("date"),
            h_prod(
                [
                    k
                    for k in df.schema.keys()
                    if k != "date"
                ],
                propagate_nulls=True,
            )
            .alias(VALUE)
            .fill_nan(None),
        )
        return df


def h_sum(
    ks: list[str],
    propagate_nulls: bool = True,
    f: Callable[[pl.Expr], pl.Expr] | None = None,
):
    return (
        pl.when(
            pl.all_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
            if not propagate_nulls
            else pl.any_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
        )
        .then(None)
        .otherwise(
            pl.sum_horizontal(
                *[
                    pl.col(k)
                    if f is None
                    else pl.col(k).pipe(f)
                    for k in ks
                ]
            )
        )
    )


def h_prod(
    ks: list[str],
    propagate_nulls: bool = True,
    f: Callable[[pl.Expr], pl.Expr] | None = None,
):
    return (
        pl.when(
            pl.all_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
            if not propagate_nulls
            else pl.any_horizontal(
                [pl.col(k).is_null() for k in ks]
            )
        )
        .then(None)
        .otherwise(
            pl.fold(
                acc=pl.lit(1.0),
                function=lambda acc, x: acc * x,
                exprs=[
                    pl.col(k).fill_nan(None)
                    if f is None
                    else pl.col(k).fill_nan(None).pipe(f)
                    for k in ks
                ],
            )
        )
    )


@dataclass(frozen=True)
class HandleNewDailyHDot(HandleNewDailyAggregate):
    dot_groups: int

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        ks = list(df.drop("date").schema.keys())
        n = self.dot_groups
        d = len(ks) / self.dot_groups
        assert d == int(d), self
        groups = [
            [ks[int((ii * d) + i)] for ii in range(n)]
            for i in range(int(d))
        ]
        df = df.select(
            pl.col("date"),
            *[
                h_prod(
                    g,
                    propagate_nulls=False,
                    f=lambda e: e.fill_null(0.0),
                )
                .alias(f"{VALUE}_{i}")
                .fill_nan(None)
                for i, g in enumerate(groups)
            ],
        ).select(
            "date",
            h_sum(
                [f"{VALUE}_{i}" for i in range(int(d))],
                propagate_nulls=True,
            )
            .alias(VALUE)
            .fill_nan(None),
        )
        return df


@dataclass(frozen=True)
class HandleNewDailyDDRegime(HandleNewDailyAggregate):
    threshold: float
    span: int
    window: int
    continuation: float
    share_dd: bool
    share_recovery: bool
    continuous: bool

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return dd_regime_df(
            rxs,
            threshold=self.threshold,
            span=self.span,
            window=self.window,
            continuation=self.continuation,
            share_dd=self.share_dd,
            share_recovery=self.share_recovery,
            continuous=self.continuous,
        )


@dataclass(frozen=True)
class HandleNewDailyMetricsEWM(HandleNewDailyAggregate):
    span: float

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return metrics_ewm_df(
            rxs,
            span=self.span,
        )


@dataclass(frozen=True)
class HandleNewDailyMetricsGrid(HandleNewDailyAggregate):
    rx_grid_width: float
    re_grid: bool

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return metrics_grid_df(
            rxs,
            rx_grid_width=self.rx_grid_width,
            re_grid=self.re_grid,
        )


@dataclass(frozen=True)
class HandleNewDailyWeightsRank(HandleNewDailyAggregate):
    metric: str
    sample: str | int
    window: float
    smooth: float
    norm: str
    top: int

    def agg_dfs(
        self,
        first: dt.date,
        last: dt.date,
        rxs: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
        return weights_rank_df(
            rxs,
            metric=self.metric,
            sample=self.sample,
            window=self.window,
            smoothing=self.smooth,
            norm=self.norm,
            top=self.top,
        )


HANDLERS = {
    "hsum": HandleNewDailyHSum,
    "hdot": HandleNewDailyHDot,
    "hprod": HandleNewDailyHProd,
    "hgross": HandleNewDailyHGross,
    "dd_regime": HandleNewDailyDDRegime,
    "metrics_ewm": HandleNewDailyMetricsEWM,
    "metrics_grid": HandleNewDailyMetricsGrid,
    "weights_rank": HandleNewDailyWeightsRank,
}


def agg_handler(
    agg: str,
    params: frozendict[str, bool | int | float | None]
    | None = None,
) -> HandleNewDailyAggregate:
    cls = assert_is(HANDLERS.get(agg))
    if params is None:
        return cls()
    try:
        return cls(**params)
    except:
        raise ValueError((cls, agg, params))


__all__ = [
    "DailyAggregate",
    "DailyAggregateDeriv",
    "get_agg_dfs",
    "agg_handler",
]
