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
    Generator,
    Callable,
)
import itertools
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
from .....apis import assert_is, t, kv
from .....engine import *

from ...calc import with_dd_regime

from ...agg import *

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
    ref: (  # TODO: rename unit? params?
        str
        | Literal["m"]
        | Literal["q"]
        | Literal["y"]
        | None
    ) = None
    threshold: tuple[float, int] | None = None

    @classmethod
    def by_year(cls):
        return cls("period", "y")

    @classmethod
    def by_quarter(cls):
        return cls("period", "q")

    @classmethod
    def by_month(cls):
        return cls("period", "m")

    def ref_label(self):
        ref = assert_is(self.ref, str)
        if "=" in ref:
            return ref.split("=")[0]
        return ref

    def ref_path(self):
        ref = assert_is(self.ref, str)
        if "=" in ref:
            return ref.split("=")[1]
        return ref

    def path_label(self) -> str:
        if self.by == "period":
            return f"partn-{self.ref}"
        elif self.by == "regime":
            return f"partn-{self.ref_label()}"
        elif self.by == "dd":
            return f"partn-dd-{self.ref_label()}"
        else:
            raise ValueError(self)

    def apply(
        self,
        df: pl.DataFrame,
        obj: DailyBtPerf,
    ) -> list[tuple[dict, pl.DataFrame]]:
        d = pl.col("date")
        df = df.with_columns(
            pl.col("rx").std().mul(16).alias("rx_sd")
        )
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
        elif self.by == "regime":
            df_reg = daily_pl_df(
                context.get().get(self.ref_path(), Daily),
                alias="regime",
            )
            df = df.join(
                df_reg,
                on="date",
                how="full",
                coalesce=True,
            ).sort(by="date")
            return df.pipe(
                apis.pl.partition_by_regime,
                date="date",
                window="window",
                regime="regime",
                with_alias=False,
                with_keys=True,
                as_list=True,
            )
        elif self.by == "dd":
            threshold, span = assert_is(self.threshold)
            df_dd = (
                daily_pl_df(
                    context.get().get(
                        self.ref_path(), Daily
                    ),
                    alias="ref",
                )
                .with_columns(
                    pl.col("ref")
                    .fill_nan(None)
                    .alias("ref")
                )
                .pipe(
                    with_dd_regime,
                    rx="ref",
                    threshold=threshold,
                    span=span,
                    alias="regime",
                    key=self.ref_path(),
                )
            )
            label = "label"
            df_dd = df_dd.with_columns(
                pl.when(pl.col(label) == 1)
                .then(pl.lit("+"))
                .when(pl.col(label) == 0)
                .then(pl.lit("~"))
                .when(pl.col(label) == -1)
                .then(pl.lit("-"))
                .otherwise(None)
                .alias(label),
            ).select(
                "date",
                pl.concat_str(
                    pl.col("regime"),
                    pl.col("label"),
                    pl.col("dd").round(2),
                    separator="|",
                )
                .backward_fill()
                .alias("regime"),
            )
            df = df.join(
                df_dd,
                on="date",
                how="full",
                coalesce=True,
            ).sort(by="date")
            # TODO: and then heatmap by only label (sign)
            return df.pipe(
                apis.pl.partition_by_regime,
                date="date",
                window="window",
                regime="regime",
                with_alias=False,
                with_keys=True,
                as_list=True,
            )
        else:
            raise ValueError(self)


def nested_setattr(d: dict | None, *keys, val=None):
    if d is None:
        d = {}
    if len(keys) == 1:
        return {**d, keys[0]: val}
    return {
        **d,
        keys[0]: nested_setattr(
            d.get(keys[0]), *keys[1:], val=val
        ),
    }


# NOTE: mutations adjust implementation parameters (including signal sign / quantile filtering)
# to plot 2d distributions of adjustment vs return (change)
# as well as the above distribution over summary stats (sharpe etc.)
# eg. signal lag, signal / scale noise, rebalance, position cap strictness
# signal quantile (for strat or bench) is mutn: long only / short only (or quantile) are strat variants not partn
@dataclass(frozen=True)
class DailyBtMutation:
    target: (
        Literal["sig"]
        | Literal["scl"]
        | Literal["reb"]
        | Literal["cap"]
        # TODO: rx?
    )
    change: (
        Literal["rel"]
        | Literal["abs"]
        | Literal["lag"]
        | Literal["levl"]
        | Literal["qile"]
        | Literal["sdev"]
    )
    params: tuple[str | float, ...] = ()

    # TODO: indicate if child or parent (and which child)
    # if mvar

    @classmethod
    def signal(cls, change: str, *params: str | float):
        return cls(
            "sig",
            change,  # type: ignore
            params,
        )

    @classmethod
    def scale(cls, change: str, *params: str | float):
        return cls(
            "scl",
            change,  # type: ignore
            params,
        )

    @classmethod
    def rebalance(cls, change: str, *params: str | float):
        return cls(
            "reb",
            change,  # type: ignore
            params,
        )

    @classmethod
    def cap(cls, change: str, *params: str | float):
        return cls(
            "cap",
            change,  # type: ignore
            params,
        )

    def expand_kwargs(self, n_samples: int = 30):

        # TODO: clip (signal, returns, etc.)

        kwargs = {}
        params = self.params
        key = {
            "sig": "signal",
            "scl": "scale",
            "reb": "rebalance",
            "cap": "cap",
        }.get(self.target, None)
        assert key is not None, self
        kwargs[key] = {}
        if self.change == "rel":
            assert self.target in ("sig", "scl", "cap"), (
                self
            )
            # TODO: len == 3
            # pass what rel to (sdev, range, etc.)
            if len(params) == 1:
                (sig,) = params
                kwargs[key]["rel_sd"] = float(sig)
            elif len(params) == 2:
                mu, sig = params
                kwargs[key]["rel_mu"] = float(mu)
                kwargs[key]["rel_sd"] = float(sig)
            else:
                raise ValueError(self)
            return [
                nested_setattr(kwargs, "sample", val=i)
                for i in range(n_samples)
            ]
        elif self.change == "abs":
            assert self.target in ("sig", "scl", "cap"), (
                self
            )
            if len(params) == 1:
                (sig,) = params
                kwargs[key]["abs_sd"] = float(sig)
            elif len(params) == 2:
                mu, sig = params
                kwargs[key]["abs_mu"] = float(mu)
                kwargs[key]["abs_sd"] = float(sig)
            else:
                raise ValueError(self)
            return [
                nested_setattr(kwargs, "sample", val=i)
                for i in range(n_samples)
            ]
        elif self.change == "lag":
            (lag,) = params
            # TODO: rand
            return [
                nested_setattr(kwargs, key, "lag", val=i)
                for i in range(int(lag))
            ]
        elif self.change == "levl":
            assert self.target in ("sig", "scl", "cap"), (
                self
            )
            if len(params) == 1:
                (sign,) = params
                if sign == "sign":
                    sep = [0.0]
                else:
                    sep = [float(sign)]
            elif len(params) == 2:
                sep = [float(v) for v in params]
            elif len(params) == 3:
                _start, _stop, _num = params
                sep = np.linspace(
                    float(_start),
                    float(_stop),
                    num=int(_num),
                ).tolist()
            acc = []
            l = None
            r = None
            for v in sep:
                acc.append(
                    {
                        key: {
                            **kwargs[key],
                            "lower": l,
                            "upper": v,
                        }
                    }
                )
                l = v
            acc.append(
                {
                    key: {
                        **kwargs[key],
                        "lower": v,
                        "upper": None,
                    }
                }
            )
            return acc
        elif self.change == "qile":
            assert self.target in ("sig", "scl", "cap"), (
                self
            )
            (qs,) = params
            kwargs[key]["quantiles"] = int(qs)
            return [
                nested_setattr(
                    kwargs, key, "quantile", val=i
                )
                for i in range(int(qs))
            ]
        elif self.change == "sdev":
            assert self.target in ("sig", "scl", "cap"), (
                self
            )
            qs, width = params
            kwargs[key]["sdevs"] = int(qs)
            kwargs[key]["sdev_width"] = float(width)
            return [
                nested_setattr(kwargs, key, "sdev", val=i)
                for i in range(int(qs))
            ]
        else:
            raise ValueError(self)
        return [kwargs]

    def path_label(self) -> str:
        assert self.change in (
            "rel",
            "abs",
            "lag",
            "levl",
            "sdev",
            "qile",
        ), self
        assert self.target in (
            "sig",
            "scl",
            "reb",
            "cap",
        ), self
        params = (
            ""
            if self.params is None or not len(self.params)
            else "-"
            + "-".join([str(p) for p in self.params])
        )
        if params.endswith(".0"):
            params = params[:-2]
        params = params.replace(".0-", "-")
        return f"mutn-{self.target}-{self.change}{params}"

    def apply(
        self,
        obj: DailyBtPerf,
        samples: int = 30,
    ) -> Generator[tuple[dict, pl.DataFrame]]:
        bt_rx = obj.get_parent()
        bt = bt_rx.get_parent()
        univ = bt.get_parent()

        if isinstance(bt, DailyBtUvar):
            kwargs: list[dict[str, dict]] = (
                self.expand_kwargs()
            )

            ((dfs, params),), d0, d1 = get_comb_dfs(
                bt.args,
                univ,
            )

            for kw in kwargs:
                kwargs_flat = {}
                for k, d in kw.items():
                    kwargs_flat = {
                        **kwargs_flat,
                        **{
                            f"{k}_{kk}": v
                            for kk, v in d.items()
                        },
                    }

                cap = params.get("cap", None)
                cap = cap if cap is None else float(cap)

                df = combine_backtest_dfs(
                    assert_is(dfs.get("rx", None)),
                    assert_is(dfs.get("scale", None)),
                    dfs.get("signal", None),
                    dfs.get("rebalance", None),
                    dfs.get("calendar", None),
                    dfs.get("weight", None),
                    dfs.get("cap", None),
                    dfs.get("fx", None),
                    cap=cap,
                    **kw,
                ).pipe(
                    run_backtest,
                    signal=None
                    if "signal" not in dfs
                    else "signal",
                    rebal=None
                    if "rebalance" not in dfs
                    else "rebalance",
                    calendar=None
                    if "calendar" not in dfs
                    else "calendar",
                    fx=None if "fx" not in dfs else "fx",
                    cap=None if "cap" not in dfs else "cap",
                    unit=1.0,
                )

                yield kwargs_flat, df

        elif isinstance(bt, DailyBtMvar):
            # TODO: basket we just mutate the outer
            # later, optional index for which child to mutate
            # then, pass multiple mutation to target several

            raise ValueError(bt)
        else:
            raise ValueError(bt)


@dataclass(frozen=True)
class DailyBtPerf(
    UseRootDataFrameStorage,
    DependsParentValue,
    Daily,  # NOTE: not chunked, store as df not series
    # TODO: not really daily? certainly cant be passed to eg. tf / rolling etc.
):
    parent: DailyBtRx | str

    bench: DailyBtRx | str | None = None
    partition: DailyBtPartition | None = None
    # TODO: partn tuple?

    mutation: DailyBtMutation | None = None
    # TODO: mutatuin tuple as well
    # TODO: then, allow target basket child, so multiople mutation can target separate children differently

    version: int = 0
    type: Literal["DailyBtPerf"] = "DailyBtPerf"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyBtPerfSnap"] = "DailyBtPerfSnap"

    def get_parent(self) -> DailyBtRx:
        return context.get().get(self.parent, DailyBtRx)

    @classmethod
    def new(
        cls,
        parent: DailyBtRx,
        bench: DailyBtRx | None = None,
        partition: DailyBtPartition | None = None,
        mutation: DailyBtMutation | None = None,
    ):
        ident = None
        if bench is not None:
            # TODO:
            # assert bench is in same domain as parent?
            # else, have to include universe ident as well?
            bench_path = obj_path(bench).replace(
                obj_path(bench.parent) + "-", ""
            )
            # TODO: some kind of shorthand aliasing on bench?
            ident = "bench-" + bench_path
        if partition is not None:
            ident = (
                partition.id_label()
                if ident is None
                else f"{ident}-{partition.id_label()}"
            )
        if mutation is not None:
            ident = (
                mutation.id_label()
                if ident is None
                else f"{ident}-{mutation.id_label()}"
            )
        if ident is None:
            ident = "perf"
        return cls(
            id=parent.extend_id(ident),
            parent=parent,
            bench=bench,
            partition=partition,
            mutation=mutation,
        )

    def depends(self) -> list[Dependency]:
        return self.depends_parent() + (
            []
            if self.bench is None
            else [
                Dependency(
                    self.id,
                    obj_path(self.bench),
                    events=("ValueUpdate",),
                )
            ]
        )

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueAppend):
            # res.append(HandleUpdateIndexDailyRx())
            raise ValueError(event)
        elif isinstance(event, ValueUpdate):
            assert self.get_parent().id == event.obj, (
                self,
                event,
            )
            return HandleNewDailyBench()
        raise ValueError((self, event))

    def display(
        self,
        row: int | None = None,
        **agg: Callable[[str], pl.Expr],
    ):
        val = self.value_key_val().get(self.id)
        if val is None:
            return
        elif val.df is None:
            return
        df = val.df
        if len(agg):
            assert "sample" in df.schema, df.schema
            it = (
                list(ks)
                for _, ks in itertools.groupby(
                    list(val.df.schema.keys()) + ["sample"],
                    key="sample".__ne__,
                )
            )
            g_keys, v_keys = [a + b for a, b in zip(it, it)]
            df = df.group_by(
                *g_keys[:-1],
                maintain_order=True,
            ).agg(
                f(k).alias(f"{k}_{kk}")
                for k in v_keys[:-1]
                for kk, f in agg.items()
            )
        if len(df) == 1 or row is not None:
            if row is None:
                row = 0
            return apis.pl.get_row(
                df, row, repr=True, round=4
            )
        return print(val.df)

    # def explorer_df(self):

    #     # TODO: call explorer df for the rx, bench, drawdown and regime

    #     return


def rx_df_to_perf_row(
    df: pl.DataFrame,
    key: str,
    partitioned: bool = False,
    **kvs,
):
    # NOTE: kvs indicate the particular mutation / partition
    return df.pipe(
        apis.perf.perf_row,
        k=key,
        rx="rx",
        pos=None,
        with_pos=False,
        with_keys=None,
        sharpe_vols=None
        if not partitioned
        else {
            "shrp": None,
            "shrp_g": "rx_sd",
        },
        # [
        #     k for k in df.schema.keys()
        #     if k not in [
        #         "date", "rx", "bench"
        #     ]
        # ],
        **kvs,
    )


@dataclass(frozen=True)
class HandleNewDailyBench(HandleNewDailyFromDailyParentPl):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyBtPerf,
    ) -> pl.DataFrame:

        rx = obj.get_parent()
        df = df.rename({VALUE: "rx"})

        assert "date" in df, (obj, df.schema)

        key = rx.id

        if obj.bench is not None:
            # TODO: add bench to key
            raise ValueError(obj)

        # TODO: date range in cols (all: =first, last rx)?
        # result rx not input rx (ie. so including is signal is only later, etc.)

        if obj.mutation is None and obj.partition is None:
            if obj.bench is not None:
                bench = context.get().get(
                    obj.bench, DailyBtRx
                )
                df = (
                    apis.pl.join_dfs(
                        [
                            df,
                            daily_pl_df(
                                bench,  # type: ignore
                                alias="bench",
                            ),
                        ],
                        on="date",
                        how="full",
                        coalesce=True,
                    )
                    .sort(by="date")
                    .with_columns(
                        pl.col("rx")
                        .sub(pl.col("bench"))
                        .alias("rx")
                    )
                )

            df = pl.DataFrame(
                [df.pipe(rx_df_to_perf_row, key)]
            )
            return df

        elif obj.mutation is None:
            assert obj.partition is not None, obj
            rs = [
                df_partn.pipe(
                    rx_df_to_perf_row,
                    key,
                    partitioned=True,
                    **partn_kvs,
                )
                for partn_kvs, df_partn in obj.partition.apply(
                    df, obj
                )
            ]

            return pl.DataFrame(
                rs, infer_schema_length=len(rs)
            )

        dfs = []
        for mutn_kvs, mutn_df in obj.mutation.apply(obj):
            if obj.partition is None:
                rs = [
                    mutn_df.pipe(
                        rx_df_to_perf_row, key, **mutn_kvs
                    )
                ]
            else:
                rs = [
                    df_partn.pipe(
                        rx_df_to_perf_row,
                        key,
                        partitioned=True,
                        **{
                            **mutn_kvs,
                            **partn_kvs,
                            # TODO: sigh annotation?
                        },
                    )
                    for partn_kvs, df_partn in obj.partition.apply(
                        mutn_df, obj
                    )
                ]
            try:
                dfs.append(
                    pl.DataFrame(
                        rs, infer_schema_length=len(rs)
                    )
                )
            except:
                raise ValueError(rs)

        return pl.concat(dfs, how="diagonal_relaxed")


mutate_signal = DailyBtMutation.signal
mutate_scale = DailyBtMutation.scale
mutate_rebalance = DailyBtMutation.rebalance
mutate_cap = DailyBtMutation.cap

partn_by_year = DailyBtPartition.by_year
partn_by_quarter = DailyBtPartition.by_quarter
partn_by_month = DailyBtPartition.by_month

# TODO: mutated rx as separate type (eg. to slice out particular quantile / signal sign as a strategy constituent itself)

__all__ = [
    "DailyBtPartition",
    "DailyBtMutation",
    "DailyBtPerf",
    "rx_df_to_perf_row",
    "HandleNewDailyBench",
    "mutate_signal",
    "mutate_scale",
    "mutate_rebalance",
    "mutate_cap",
    "partn_by_year",
    "partn_by_quarter",
    "partn_by_month",
]
