
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

from .core import *
from .bt import *

#  ------------------

# NOTE: the relevant bench eg. could be without signal, but same scale on the same asset (prob post bt vol rescaled)

# but flexible for you to use to see whatever relative perf you're interested in

# eg. could be a base-case signal (eg. trend with or without a stress modulator on speed)

# or industrial sector specific signal vs overall (eg. cap util, ind pro)

#  ------------------

# up to the user to bind whichever partn/mutn they're interested in for a given project

#  ------------------

# NOTE: partitions group logically meaningful consecutive periods
# to use for bootstrapping distributions over summary stats
# eg. distribution of possible sharpe, annual rx, vol, etc.
# eg. year, regime, underlying dd period
@dataclass(frozen=True)
class DailyBtPartition:
    by: (
        Literal["period"]
        # TODO: period rule? ie. 3rd week or whatever
        | Literal["regime"]
        | Literal["dd"]
    )
    ref: (
        int 
        | Literal["m"] 
        | Literal["q"] 
        | Literal["y"] 
        | None
    ) = None

    def path_label(self) -> str:
        if self.by == "period":
            return f"partn-{self.ref}"
        elif self.by == "regime":
            return f"partn-" # TODO:
        elif self.by == "dd":
            return f"partn-" # TODO:
        else:
            raise ValueError(self)
    
    def apply(
        self,
        df: pl.DataFrame,
        obj: DailyBtBench,
        domain: Domain,
    ) -> list[
        tuple[dict, pl.DataFrame]
    ]:
        d = pl.col("date")
        if self.by == "period" and self.ref == "m":
            return df.pipe(
                apis.pl.partition_by_exprs,
                with_alias=False,
                with_keys=True,
                as_list=True,
                year=d.dt.year(),
                month=d.dt.month(),
            )
        if self.by == "period" and self.ref == "q":
            return df.pipe(
                apis.pl.partition_by_exprs,
                with_alias=False,
                with_keys=True,
                as_list=True,
                year=d.dt.year(),
                month=d.dt.month().mod(3),
            )
        if self.by == "period" and self.ref == "y":
            return df.pipe(
                apis.pl.partition_by_exprs,
                with_alias=False,
                with_keys=True,
                as_list=True,
                year=d.dt.year(),
            )
        elif self.by:
            raise ValueError(self)
        elif self.by == "regime":
            # TODO: get ref series
            # partition by continuous periods of same val
            # round? quantile? (no, do that user side)
            return []
        elif self.by == "dd":
            # TODO: get ref series
            # identify dd periods
            # partition by continuous period of same y/n dd
            return []
        else:
            raise ValueError(self)

# NOTE: mutations adjust implementation parameters (including signal sign / quantile filtering)
# to plot 2d distributions of adjustment vs return (change)
# as well as the above distribution over summary stats (sharpe etc.)
# eg. signal lag, signal / scale noise, rebalance, position cap strictness
# signal quantile (for strat or bench) is mutn: long only / short only (or quantile) are strat variants not partn
@dataclass(frozen=True)
class DailyBtMutation:
    target: (
        Literal["signal"]
        | Literal["scale"]
        | Literal["rebalance"]
        | Literal["cap"]
    )
    # change: quantile, noise, shift
    # levels: list[float] | None = None

    # if indicated to do so, apply to the asset level return series
    # as well as the overall

    def path_label(self) -> str:
        return ""
        
    def apply(
        self,
        obj: DailyBtBench,
        domain: Domain,
    ) -> list[
        tuple[dict, pl.DataFrame]
    ]:
        bt_rx = obj.get_parent(domain)
        bt = bt_rx.get_parent(domain)
        univ = bt.get_parent(domain)
        if isinstance(bt, DailyBtUvar):
            (
                df_rx,
                df_scale,
                df_signal,
                df_rebalance,
                df_calendar,
                df_weight,
                df_cap,
                df_fx,
            ) = get_backtest_dfs(
                univ.bt[bt.index], domain, univ
            )

        # generate mutations
        # change
        # re run backtest

        # return df and mutation keys, per variation
        # TODO: change to generator

        return []

    
@dataclass(frozen=True)
class DailyBtBench(
    DependsParentValue,
    Daily, # NOTE: not chunked, store as df not series
):
    parent: DailyBtRx | str
    
    bench: DailyBtRx | str | None = None
    partition: DailyBtPartition | None = None
    # TODO: partn tuple

    mutation: DailyBtMutation | None = None
    # TODO: mutatuin tuple as well
    
    version: int = 0
    type: Literal["DailyBtBench"] = "DailyBtBench"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtBenchSnap"] = "DailyBtBenchSnap"
 
    def get_snap(
        self,
        domain: Domain,
        t: Type[DailySnap] | None = None,
    ) -> DailyBtBench.Snap | None:
        return self.snap_key_val(domain).get(self.path, DailyBtBench.Snap)
    
    def get_parent(self, domain: Domain) -> DailyBtRx:
        return domain.get(self.parent, DailyBtRx)
    
    @classmethod
    def new(
        cls,
        parent: DailyBtRx,
        bench: DailyBtRx | None = None,
        partition: DailyBtPartition | None = None,
        mutation: DailyBtMutation | None = None,
    ):
        ident = "bench"
        if bench is not None:
            # TODO:
            # assert bench is in same domain as parent?
            # else, have to include pfo ident as well?
            bench_path = obj_path(bench).replace(
                obj_path(bench.parent) + "-", ""
            )
            ident += "-" + bench_path
        if partition is not None:
            ident += partition.path_label()
        if mutation is not None:
            ident += mutation.path_label()
        return cls(
            path=parent.extend_path(ident),
            parent=parent,
            bench=bench,
            partition=partition,
            mutation=mutation,
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_parent(domain)
            + (
                []
                if self.bench is None
                else [
                    Dependency(
                        self.path,
                        obj_path(self.bench),
                        events=("ValueUpdate",)
                    )
                ]
            )
        )

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, ValueAppend):
            # res.append(HandleUpdateIndexDailyRx())
            raise ValueError(event)
        elif isinstance(event, ValueUpdate):
            assert self.get_parent(domain).path == event.obj, (self, event)
            res.append(HandleNewDailyBench())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

    def explorer_df(self, domain: Domain):

        # TODO: call explorer df for the rx, bench, drawdown and regime

        return

@dataclass(frozen=True)
class HandleNewDailyBench(
    HandleNewDailyFromDailyParentPl
):
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: DailyBtBench,
        domain: Domain,
    ) -> pl.DataFrame:
        df = df.rename({obj.path: "rx"})
        
        if obj.mutation is None:

            if obj.bench is not None:
                bench = domain.get(obj.bench, DailyBtRx)
                df = apis.pl.join_dfs(
                    df,
                    daily_pl_df(bench, domain, alias="bench"),
                    on="date",
                    how="full",
                    coalesce=True,
                ).sort(by="date")

                # TODO: rx is spread? any other change?

            if obj.partition is None:
                df = pl.DataFrame([
                    df.pipe(
                        apis.perf.perf_row,
                        k=obj.path,
                        rx="rx",
                        pos=None,
                        with_pos=False,
                        with_keys = [
                            k for k in df.schema.keys()
                            if k not in [
                                "date", "rx", "bench"
                            ]
                        ]
                    )
                ])
            else:
                df = pl.DataFrame([
                    df_partn.pipe(
                        apis.perf.perf_row,
                        k=obj.path,
                        rx="rx",
                        pos=None,
                        with_pos=False,
                        with_keys = [
                            k for k in df.schema.keys()
                            if k not in [
                                "date", "rx", "bench"
                            ]
                        ],
                        **partn_kvs
                    )
                    for partn_kvs, df_partn
                    in obj.partition.apply(
                        df, obj, domain
                    )
                ])

            return df
        
        df = pl.concat([
            self.calc_df(
                mutn_df.with_columns(
                    **mutn_kvs
                ), event, obj, domain
            )
            for mutn_kvs, mutn_df in obj.mutation.apply(
                obj, domain
            )
        ], how = "diagonal_relaxed")

        return df
            