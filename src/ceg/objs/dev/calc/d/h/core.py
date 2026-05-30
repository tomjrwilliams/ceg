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

from ......apis.calcs.metrics import *
from ......apis.calcs.weights import *
from ......apis.calcs.regimes.dd import *

from .rank import *
from .quantize import *
from .trans_prob import *


def _h_sum(
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
def _h_prod(
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

# def get_composition_dfs(
#     refs: tuple[Ref | str, ...],
#     g: Graph,
# ) -> tuple[list[pl.DataFrame], dt.date, dt.date]:
#     """Resolve refs to Daily objects, join their dfs, return [single df], d_min, d_max.

#     Each ref contributes one column (named by the resolved Object's id) to a
#     single joined df. Compositions over multiple groups should be modelled as
#     sub-Containers, each of which is itself a ref.
#     """
#     if not len(refs):
#         raise ValueError("DailyComposition.refs is empty")

#     objs = [assert_is(g.get(r, Object), Daily) for r in refs]

#     df = (
#         apis.pl.join_dfs(
#             [daily_pl_df(o) for o in objs],
#             on="date",
#             align=True,
#         )
#         .sort(by="date")
#         .select("date", *[o.id for o in objs])
#     )

#     d0, d1 = apis.pl.select_items(
#         df,
#         (pl.col("date").min(), pl.col("date").max()),
#         dt.date,
#     )
#     return [df], d0, d1


    # dfs, d0, d1 = get_composition_dfs(obj.refs, g)
    # dfs = dfs + [
    #     daily_pl_df(assert_is(g.get(p, Object), Daily)) for p in obj.peers
    # ]

    # if snap.first and snap.last:
    #     if d0 == snap.first and d1 == snap.last:
    #         return frozendict().set(obj.id, (snap, ()))

    # res = self.agg_dfs(d0, d1, *dfs)

    # df = None
    # snaps = frozendict()

    # else:
    #     raise ValueError((type(res), self))

class DailyHChild(Daily, HasDataFrameValue):
    pass

@dataclass(frozen=True)
class DailyH(
    Daily,
    TriageUpstream,
    HasDataFrameValue
):
    # TODO: dot could probably benefit from being two lists of inputs
    # TODO: ie in future migth have H1, H2 etc. for how many input params?

    @dataclass(frozen=True)
    class Snap(TriageUpstream.Snap, Daily.Snap):
        triage: bool = True

    @dataclass(frozen=True)
    class Params(Daily.Params):
        func: str
        kwargs: t.frozen.dict[str, bool | int | float | str] = frozendict()

        @classmethod
        def sum(cls, **kwargs: bool | int | float | str):
            return cls("sum", frozendict(kwargs))
        
        @classmethod
        def rank(cls, **kwargs: bool | int | float | str):
            return cls("rank", frozendict(kwargs))
        
        @classmethod
        def quantize(cls, **kwargs: bool | int | float | str):
            return cls("quantize", frozendict(kwargs))
        
        @classmethod
        def transition_prob(cls, **kwargs: bool | int | float | str):
            return cls("transition_prob", frozendict(kwargs))
        
        @classmethod
        def dot(cls, **kwargs: bool | int | float | str):
            return cls("dot", frozendict(kwargs))
            
        @classmethod
        def prod(cls, **kwargs: bool | int | float | str):
            return cls("prod", frozendict(kwargs))
            
        @classmethod
        def gross(cls, **kwargs: bool | int | float | str):
            return cls("gross", frozendict(kwargs))
            
        @classmethod
        def dd_regime(cls, **kwargs: bool | int | float | str):
            return cls("dd_regime", frozendict(kwargs))
            
        @classmethod
        def metrics_ewm(cls, **kwargs: bool | int | float | str):
            return cls("metrics_ewm", frozendict(kwargs))
            
        @classmethod
        def metrics_grid(cls, **kwargs: bool | int | float | str):
            return cls("metrics_grid", frozendict(kwargs))
            
        @classmethod
        def weights_rank(cls, **kwargs: bool | int | float | str):
            return cls("weights_rank", frozendict(kwargs))
            

    parent: Object | str
    params: Params

    inputs: tuple[IdRef | AliasRef | MemberRef, ...]

    def self(self) -> DailyH:
        return self
    
    def encode_type(self, params: DailyH.Params):
        return "h"

    def encode_params(self, params: DailyH.Params):
        tags = [params.func]
        for k, v in params.kwargs.items():
            tags.append(f"{k}-{v}")
        return "-".join(map(str, tags))
    
    def depends_fields(self):
        return {
            "inputs": ("ValueUpdate",)
        }

    def handle(
        self,
        event: Event,
        snap: Snap,
    ):
        obj = self
        params = self.params

        event = assert_is(event, ValueUpdate)
        snap = assert_is(snap, self.Snap)

        d1, d0 = assert_is(snap.last, dt.date), assert_is(snap.first, dt.date)

        g = context.get()
        inputs = [
            g.get(p, Daily) for p in self.inputs
        ]
        
        df = daily_pl_df(inputs)

        res = AGGREGATES[params.func](
            df,
            obj.id,
            inputs=inputs,
            kwargs = params.kwargs,
        )
        print("HANDLE AGG")

        snaps = frozendict()

        # fire_every_time in kwargs?
        if isinstance(res, pl.DataFrame):
            df = res
            assert len(df) == ((d1 - d0).days + 1), (df, d1, d0)
            self.set_value_from_df(
                df, self.id,
                # or VALUE
            )
        elif isinstance(res, tuple):
            df, k_is = res
            assert len(df) == ((d1 - d0).days + 1), (df, d1, d0)
            _snaps = {}

            def store_child(k_i):
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
                child = obj.deriv(key=k, index=i)
                context.get().with_obj(child, force=True)
                child_snap = get_snap(child)

                child.set_value_from_df(df, col)
                
                _snaps[child.id] = (
                    replace(child_snap, first=d0, last=d1),
                    ValueUpdate(child.id),
                )

            list(map(store_child, k_is))
            for k, v in _snaps.items():
                assert k not in snaps, k
                snaps = snaps.set(k, v)

        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(df),
            into=snaps,
        )
    
    def deriv(self, key: str | None, index: int | None) -> DailyHChild:
        raise NotImplementedError(self)

def aggregate_sum(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    # if isinstance(inputs, frozendict):
    #     params = tuple(inputs.values())
    # return df.select(
    #     "date",
    #     pl.sum_horizontal(*[
    #         pl.col(i.id) for i in inputs
    #     ]).alias(alias)
    # )
    ids = [i.id for i in inputs]
    return df.select(
        pl.col("date"),
        _h_sum(
            ids,
            propagate_nulls=False,
        )
        .alias(VALUE)
        .fill_nan(None),
    )


def aggregate_gross(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    ids = [i.id for i in inputs]
    df = df.select(
        pl.col("date"),
        _h_sum(
            ids,
            f=lambda e: e.abs().fill_null(0.0),
            propagate_nulls=False,
        )
        .alias(VALUE)
        .fill_nan(None),
    )
    return df


def aggregate_prod(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    ids = [i.id for i in inputs]
    df = df.select(
        pl.col("date"),
        _h_prod(
            ids,
            propagate_nulls=True,
        )
        .alias(VALUE)
        .fill_nan(None),
    )
    return df


def aggregate_dot(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    dot_groups = assert_is(kwargs.get("dot_groups"), int)
    ids = [i.id for i in inputs]
    ks = ids
    n = dot_groups
    d = len(ks) / dot_groups
    assert d == int(d), (df, alias)
    groups = [
        [ks[int((ii * d) + i)] for ii in range(n)]
        for i in range(int(d))
    ]
    df = df.select(
        pl.col("date"),
        *[
            _h_prod(
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
        _h_sum(
            [f"{VALUE}_{i}" for i in range(int(d))],
            propagate_nulls=True,
        )
        .alias(VALUE)
        .fill_nan(None),
    )
    return df

def aggregate_dd_regime(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    return dd_regime_df(
        df,
        threshold=assert_is(kwargs.get("threshold"), float),
        span=assert_is(kwargs.get("span"), int),
        window=assert_is(kwargs.get("window"), int),
        continuation=assert_is(kwargs.get("continuation"), float),
        share_dd=assert_is(kwargs.get("share_dd"), bool),
        share_recovery=assert_is(kwargs.get("share_recovery"), bool),
        continuous=assert_is(kwargs.get("continuous"), bool),
    )


def aggregate_metrics_ewm(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    return metrics_ewm_df(
        df,
        span=assert_is(kwargs.get("span"), int),
    )


def aggregate_metrics_grid(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    return metrics_grid_df(
        df,
        rx_grid_width=assert_is(kwargs.get("rx_grid_width"), float),
        re_grid=assert_is(kwargs.get("re_grid"), bool),
    )


def aggregate_weights_rank(
    df: pl.DataFrame,
    alias: str,
    inputs: list[Daily],
    kwargs: frozendict[str, bool | int | float | str],
):
    return weights_rank_df(
        df,
        metric=assert_is(kwargs.get("metric"), str),
        sample=assert_is(kwargs.get("sample")), # type: ignore
        # , str | int
        window=assert_is(kwargs.get("window"), float),
        smoothing=assert_is(kwargs.get("smooth"), float),
        norm=assert_is(kwargs.get("norm"), str),
        top=assert_is(kwargs.get("top"), int),
    )



AGGREGATES = {
    "sum": aggregate_sum,
    "dot": aggregate_dot,
    "prod": aggregate_prod,
    "gross": aggregate_gross,
    "dd_regime": aggregate_dd_regime,
    "metrics_ewm": aggregate_metrics_ewm,
    "metrics_grid": aggregate_metrics_grid,
    "weights_rank": aggregate_weights_rank,
    "rank": aggregate_rank,
    "quantize": aggregate_quantize,
    # "sign_code": sign_code,  # TODO: lives in app/dev/states/sign_strat.py — layering inversion to import here
    "transition_prob": aggregate_transition_probs,
}

h_sum = new(DailyH, DailyH.Params.sum)
h_dot = new(DailyH, DailyH.Params.dot)
h_prod = new(DailyH, DailyH.Params.prod)
h_gross = new(DailyH, DailyH.Params.gross)
h_dd_regime = new(DailyH, DailyH.Params.dd_regime)
h_metrics_ewm = new(DailyH, DailyH.Params.metrics_ewm)
h_metrics_grid = new(DailyH, DailyH.Params.metrics_grid)
h_weights_rank = new(DailyH, DailyH.Params.weights_rank)
h_rank = new(DailyH, DailyH.Params.rank)
h_quantize = new(DailyH, DailyH.Params.quantize)
h_transition_prob = new(DailyH, DailyH.Params.transition_prob)


__all__ = [
    "DailyH",
    "h_sum",
    "h_rank",
    "h_quantize",
    "h_transition_prob",
]