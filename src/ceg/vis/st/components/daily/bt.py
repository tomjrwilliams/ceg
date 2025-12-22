from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Type,
    Any,
    Callable,
    Literal,
    overload,
    cast,

)
from pathlib import Path
import datetime as dt
import numpy as np
import polars as pl
import streamlit as st
import plotly.express as px

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...variables import *
from ..core import *
from ..dates import *
from ..dfs import *

from ..... import apis
from .....apis import t
from .....apis import perf

from .explorer import *
    
@dataclass(frozen=True)
class DailyBackTest(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    # TODO: key for which of signal, scale, etc.

    rebal: str = "daily"

    def render(self) -> DailyBackTest:
        
        df = self.df
        if not len(df):
            return self

        exp = self.explorer
        df = self.render_date_filter(df)

        if exp.group_key in df.schema:
            df_groups = {
                g: g_df.pivot(
                    on=exp.label_key,
                    index=exp.date_key,
                    values=exp.value_key,
                    # order?
                )
                for (g,), g_df in df.partition_by(
                    exp.group_key,
                    as_dict=True,
                ).items()
            }
        else:
            df_groups = {
                "rx": df.pivot(
                    on=exp.label_key,
                    index=exp.date_key,
                    values=exp.value_key,
                    # order?
                )
            }

        df_rx = df_groups.pop("rx", None)
        assert df_rx is not None, list(
            df_groups.keys()
        )

        ks = [
            k for k in df_rx.columns if k != exp.date_key
        ]
        bt_dfs = []

        for k in ks:
            k_pos, k_rx = f"{k}_pos", f"{k}_rx"
            bt_df = df_rx.select(
                exp.date_key,
                pl.col(k),
                pl.col(k)
                .is_null()
                .not_()
                .alias("calendar")
            )
            g_kwargs = {}
            for g, g_df in df_groups.items():
                
                # TODO: permutations of sig and scale
                # or just scale only

                # include_sig = only or both

                if k not in g_df.schema:
                    continue
                bt_df = bt_df.join(
                    g_df.select(
                        exp.date_key,
                        pl.col(k).pipe(
                            lag_given_group, g
                        ).alias(g),
                    ),
                    on=exp.date_key,
                    how="left",
                )
                g_kwargs[g] = g

            bt_df = bt_df.pipe(
                perf.with_backtest,
                (k_pos, k_rx),
                cum=False,
                rx=k,
                calendar="calendar",
                **g_kwargs,
                unit=1,
            )
            bt_dfs.append(bt_df)
        return self.calc(df_rx, bt_dfs, ks, exp)
    
    def calc(
        self,
        df_rx: pl.DataFrame,
        df_bts: list[pl.DataFrame],
        ks: list[str],
        exp: ExploreDaily,
    ) -> DailyBackTest: ...

def lag_given_group(e: pl.Expr, g: str) -> pl.Expr:
    if g == "scale":
        e_res = e.shift(1, fill_value=0)
    elif g == "signal":
        e_res = e.shift(1, fill_value=0)
    else:
        return e
    e_res = e_res.forward_fill()
    return pl.when(e.is_null()).then(None).otherwise(e_res)

@dataclass(frozen=True)
class DailyBackTestLineRx(DailyBackTest):

    def calc(
        self,
        df_rx: pl.DataFrame,
        df_bts: list[pl.DataFrame],
        ks: list[str],
        exp: ExploreDaily,
    ):
        df = pl.concat((
            df_rx.select(exp.date_key),
            *([
                df_bt.select(
                    pl.col(f"{k}_rx").alias(k)
                )
                for k, df_bt in zip(ks, df_bts)
            ])
        ), how="horizontal").unpivot(
            index=exp.date_key,
            on=ks,
            variable_name="label",
            value_name="rx",
        )

        fig = px.line(
            df,
            x=exp.date_key,
            y="rx",
            color="label",
        )
        st.plotly_chart(fig)

        return self

@dataclass(frozen=True)
class DailyBackTestTableRx(DailyBackTest):

    include_sig: str = "only"

    @classmethod
    def parse_params(cls, s: str | None) -> dict:
        acc = {}
        if s is None:
            return acc
        for kv in s.split(","):
            kv = kv.strip()
            k, v = kv.split("=")
            acc[k] = v
        return {
            # cum": ...,
            # rebal: ...,

        }

    def calc(
        self,
        df_rx: pl.DataFrame,
        df_bts: list[pl.DataFrame],
        ks: list[str],
        exp: ExploreDaily,
    ):
        res = []

        # NOTE: we use pct=False for hwm
        # if RxCum, then use pct

        for k, bt_df in zip(ks, df_bts):

            # TODO: from the k, bt_df
            # should identify if sig and scale
            # or just scale

            # and return no sig as well if relevant

            # ie. we optionally plot a separate variant with just scale
            # only signal flag (as always scale)
            r = perf.perf_row(
                bt_df,
                k=k,
                sig=True,
            )
            res.append(r)

        st.dataframe(pl.DataFrame(res))

        return self

EXPLORER_COMPONENTS["bt_line_rx"] = DailyBackTestLineRx
EXPLORER_COMPONENTS["bt_table"] = DailyBackTestTableRx

__all__ = [
    "DailyBackTest",
    "lag_given_group",
    "DailyBackTestLineRx",
    "DailyBackTestTableRx",
]