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

import matplotlib.pyplot as mpl

import plotly.express as px
import plotly.graph_objects as go

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

st = apis.st.library()

from .....apis import assert_is, t, perf

from .explorer import *


@dataclass(frozen=True)
class DailyDDRegimes(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    n: int = 2
    width: float = 0.1

    def render(self) -> DailyDDRegimes:

        df = self.df
        if not len(df):
            return self

        exp = self.explorer
        # df = self.render_date_filter(df)

        ks = list(df.schema.keys())

        df_rx = df.filter(group="rx").with_columns(
            pl.col(exp.value_key)
            .fill_nan(None)
            .fill_null(0)
            .cum_sum()
            .forward_fill()
            .over(exp.label_key)
        )

        v_min, v_max = df_rx.pipe(
            apis.pl.select_items,
            (
                pl.col(exp.value_key).min(),
                pl.col(exp.value_key).max(),
            ),
            t=float,
        )
        v_rng = v_max - v_min
        v_w = v_rng * self.width

        n = self.n
        v_grid = [i * v_w for i in range(n)]
        v_ex = n * v_w

        df_regime = (
            df.filter(group="regime").filter(
                pl.col(exp.value_key).is_not_null()
            )
        ).join(
            df_rx.rename({exp.value_key: "_rx"}),
            on=["date", exp.label_key],
            how="left",
        )

        rs_jitter = []
        rs_regime = []

        REGIMES = {
            1: "high",
            0.5: "rec",
            0: "chg",
            -0.5: "dd",
            -1: "low",
        }

        colors = px.colors.qualitative.Plotly
        labels = {}

        for r in df_regime.iter_rows(named=True):
            d = r["date"]
            label = r[exp.label_key]
            regime = r[exp.value_key]
            rx = r["_rx"]

            if label not in labels:
                labels[label] = colors[
                    len(labels) % len(colors)
                ]

            try:
                reg = REGIMES[regime]
            except:
                raise ValueError((regime, REGIMES))

            rs_regime.extend(
                [
                    {
                        "date": d,
                        "group": label,
                        exp.label_key: (
                            f"{label}-{REGIMES[regime]}"
                        ),
                        exp.value_key: (
                            rx + vx
                            if regime > 0
                            else rx - vx
                            if regime < 0
                            else rx + vx
                        ),
                    }
                    for vx in (
                        [v_ex]
                        if regime != 0
                        else [v_ex, -v_ex]
                    )
                ]
            )

            rs_jitter.extend(
                [
                    {
                        "date": d,
                        "group": label,
                        exp.label_key: label,
                        exp.value_key: (
                            rx + v_jit
                            if regime > 0
                            else rx - v_jit
                            if regime < 0
                            else rx
                        ),
                    }
                    for v_jit in v_grid
                ]
            )

        df_jitter = pl.DataFrame(rs_jitter)

        fig = go.Figure()

        # TODO: drop the non event day rx

        for (lg,), df in df_rx.partition_by(
            exp.label_key, as_dict=True
        ).items():
            for t in (
                px.line(
                    df,
                    # df.join(
                    #     df_jitter,
                    #     on=["date", exp.label_key],
                    #     how="semi"
                    # ),
                    x=exp.date_key,
                    y=exp.value_key,
                    color=exp.label_key,
                    color_discrete_map={lg: labels[lg]},
                )
                .update_traces(legendgroup=lg)
                .data
            ):
                fig.add_trace(t)

        for (lg,), df in df_jitter.partition_by(
            "group", as_dict=True
        ).items():
            for t in (
                px.scatter(
                    df,
                    x="date",
                    y=exp.value_key,
                    color=exp.label_key,
                    color_discrete_map={lg: labels[lg]},
                )
                .update_traces(
                    legendgroup=lg,
                )
                .data
            ):
                fig.add_trace(t)

        for (lg,), df in (
            pl.DataFrame(rs_regime)
            .partition_by("group", as_dict=True)
            .items()
        ):
            for t in (
                px.scatter(
                    df,
                    x="date",
                    y=exp.value_key,
                    color=exp.label_key,
                    color_discrete_map={
                        f"{lg}-high": "green",
                        f"{lg}-rec": "blue",
                        # f"{lg}-chg" : "blue",
                        f"{lg}-dd": "yellow",
                        f"{lg}-low": "red",
                    },
                )
                .update_traces(legendgroup=lg)
                .data
            ):
                fig.add_trace(t)

        st.plotly_chart(fig)

        return self


EXPLORER_COMPONENTS["dd_regimes"] = DailyDDRegimes

__all__ = ["DailyDDRegimes"]
