from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Type,
    Any,
    Callable,
    Literal,
    ParamSpec,
    Concatenate,
    Iterable,
    overload,
    cast,
    get_origin,
)
from pathlib import Path
import datetime as dt
import numpy as np
import math
import polars as pl
import functools

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
from .utils import *

from .....apis.calcs.weights.dd import calc_rolling_dd

import numba as nb

WINDOWS = {"M": 31, "Q": 90, "Y": 365}


def rolling_dd_(
    v_rx: np.ndarray,
    v_null: np.ndarray,
    v_sample: np.ndarray,
    window: int,
    n_t: int,
):
    try:
        return calc_rolling_dd(
            v_rx,
            v_null,
            v_sample,
            window,
            n_t,
        )
    except:
        raise ValueError(
            (
                v_rx[:3],
                v_null[:3],
                v_sample[:3],
                window,
                n_t,
            )
        )


def fill_inf(e: pl.Expr, v):
    return pl.when(e.is_infinite()).then(v).otherwise(e)


def window_rx(
    k: str,
    window: str,
    ann: float = 252.0,
):
    e = pl.col(k)
    e_mu = (
        e.rolling_mean(WINDOWS[window], min_samples=2) * ann
    )
    return e_mu.pipe(fill_inf, None).alias(k)


def window_sharpe(
    k: str,
    window: str,
    ann: float = 252.0,
):
    e = pl.col(k)
    e_mu = (
        e.rolling_mean(WINDOWS[window], min_samples=2) * ann
    )
    e_sig = e.rolling_std(
        WINDOWS[window], min_samples=2
    ) * math.sqrt(ann)
    return (
        (e_mu / e_sig)
        .fill_nan(None)
        .pipe(fill_inf, None)
        .alias(k)
    )


def downsample(
    df: pl.DataFrame,
    sample: str,
):
    if sample == "M":
        return (
            df.group_by(
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
                maintain_order=True,
            )
            .agg(
                pl.col("date").last(),
                *(
                    pl.col(k)
                    .filter(pl.col(k).is_not_null())
                    .last()
                    for k in df.schema.keys()
                    if k != "date"
                ),
            )
            .drop("year", "month")
        )
    raise ValueError(df)


def render_metric_violin(
    df: pl.DataFrame,
    chart_key: str,
    field: str,
    metric: str,
    labels: list[LabelOsc | LabelWm | LabelConst],
    sample: str = "M",
    window: str = "M",
):
    symbols = list(
        sorted(set([label.symbol for label in labels]))
    )

    if field == "span":
        spans = list(
            sorted(set([label.span for label in labels]))
        )
        col_opts = {"symbol": symbols, "span": spans}
        sels = {}
        for col, opts in col_opts.items():
            var = StateVariable(f"{chart_key}_{col}_sel")
            sels[col] = set(
                st.segmented_control(
                    col,
                    opts,
                    selection_mode="multi",
                    key=var.key,
                )
            )

        labels = [
            label
            for label in labels
            if label.symbol in sels["symbol"]
            and label.span in sels["span"]
        ]

        if not len(labels):
            return

    else:
        raise ValueError(field)

    charts = set(
        st.segmented_control(
            "chart",
            ["line", "violin"],
            selection_mode="multi",
            key=f"{chart_key}_chart_sel",
        )
    )

    if metric == "sharpe":
        df = df.with_columns(
            *(
                window_sharpe(
                    label.col,
                    window,
                ).alias(label.col)
                for label in labels
            )
        ).pipe(downsample, sample)
    elif metric == "calmar" or metric == "dd":
        df_mask = df.pipe(
            apis.pl.with_sample_masks,
            sample,
            [k for k in df.schema.keys() if k != "date"],
        )
        df = (
            df_mask.with_columns(
                *(
                    pl.Series(
                        f"{label.col}_dd",
                        rolling_dd_(
                            *apis.pl.to_numpy_with_null(
                                df_mask.get_column(
                                    label.col
                                )
                            ),
                            df_mask.get_column(
                                f"{label.col}_{sample}"
                            ).to_numpy(),
                            WINDOWS[window],
                            len(df),
                        ),
                    )
                    .fill_nan(None)
                    .forward_fill()
                    .alias(f"{label.col}_dd")
                    for label in labels
                )
            )
            .select(
                "date",
                *(
                    (
                        window_rx(
                            label.col,
                            window,
                        )
                        / pl.col(f"{label.col}_dd")
                    )
                    .fill_nan(None)
                    .pipe(fill_inf, None)
                    .alias(label.col)
                    for label in labels
                ),
            )
            .pipe(downsample, sample)
        )
    else:
        raise ValueError(metric)

    # df_state = df.pipe(
    #     radio_filter_df,
    #     [
    #         "symbol",
    #         "span",
    #         "field",
    #     ],
    #     self.key,
    # )

    df_tall = (
        df.unpivot(
            on=[label.col for label in labels],
            index="date",
            value_name=metric,
            variable_name="strat",
        )
        .filter(pl.col(metric).is_not_null())
        .with_columns(
            pl.col("strat")
            .replace(
                {
                    label.col: label.symbol
                    for label in labels
                }
            )
            .alias("symbol"),
            pl.col("strat")
            .replace(
                {
                    label.col: label.name_without(field)
                    for label in labels
                }
            )
            .alias("group"),
        )
    )

    # TODO: add selection filters for the other fields
    # eg. re entry, thresholding etc. for wm

    if "line" in charts:
        fig_line = go.Figure()
        for (group,), df_short in df_tall.partition_by(
            "group", as_dict=True
        ).items():
            for t in (
                px.line(
                    df_short.with_columns(
                        pl.col(metric)
                        .mul(1 / 100)
                        .cum_sum()
                        .over("strat")
                        .alias(metric)
                    ),
                    x="date",
                    y=metric,
                    color="strat",
                )
                .update_traces(
                    legendgroup=group,
                    legendgrouptitle_text=group,
                )
                .data
            ):
                fig_line.add_trace(t)

        st.plotly_chart(
            fig_line.update_layout(
                legend=dict(groupclick="toggleitem")
            ),
            key=f"{chart_key}_ts",
        )

    if field == "span" and "violin" in charts:
        df_tall = df_tall.with_columns(
            pl.col("strat")
            .replace(
                {label.col: label.span for label in labels}
            )
            .alias(field)
        )

        fig_viol = go.Figure()
        for t in px.violin(
            df_tall, x="span", y=metric, color="group"
        ).data:
            fig_viol.add_trace(t)
        for t in px.line(
            df_tall.group_by(
                "group",
                "span",
            )
            .agg(pl.col(metric).median().alias(metric))
            .sort("group", "span"),
            x="span",
            y=metric,
            color="group",
        ).data:
            fig_viol.add_trace(t)

        st.plotly_chart(
            fig_viol.update_layout(
                legend=dict(groupclick="toggleitem")
            ),
            key=f"{chart_key}_violin",
        )

    return


@dataclass(frozen=True)
class Label:
    col: str


@dataclass(frozen=True)
class LabelMisc(Label):
    pass


@dataclass(frozen=True)
class LabelConst(Label):
    symbol: str
    variant: str
    span: int = 0

    def name_without(self, *keys: str):
        return self.col


@dataclass(frozen=True)
class LabelOsc(Label):
    symbol: str
    span: int
    variant: str
    norm: str

    def name_without(self, *keys: str):
        ks = set(keys)
        s = f"{self.symbol}_osc"
        for k in [
            "span",
            "variant",
            "norm",
        ]:
            if k not in ks:
                s += f"_{getattr(self, k)}"
        return s.replace("-", "n").replace(".0", "")


@dataclass(frozen=True)
class LabelWm(Label):
    symbol: str
    span: int
    threshold: float
    reentry: float | None
    variant: str
    norm: str

    def name_without(self, *keys: str):
        ks = set(keys)
        s = f"{self.symbol}_wm"
        for k in [
            "span",
            "threshold",
            "reentry",
            "variant",
            "norm",
        ]:
            if k not in ks:
                s += f"_{getattr(self, k)}"
        return s.replace("-", "n").replace(".0", "")


def parse_negatable(v: str):
    if v[0].lower() == "n":
        return -1 * float(v[1:])
    else:
        return float(v)


L = TypeVar("L", bound=Label)


@functools.lru_cache(maxsize=1024)
def parse_label(k: str, t: Type[L]) -> L:
    if k == "date":
        return assert_is(LabelMisc(k), t)
    split = k.split("_")
    if len(split) < 2:
        symbol, variant = k.split("-", maxsplit=1)
        try:
            return assert_is(
                LabelConst(k, symbol, variant), t
            )
        except:
            raise ValueError(
                (
                    t,
                    get_origin(t),
                )
            )
    symbol, sig, *_ = split
    if sig == "osc":
        symbol, _, span, rest = split
        split = rest.split("-")
        if len(split) == 1:
            (variant,) = split
        elif len(split) == 2:
            variant, norm = split
        else:
            raise ValueError(k)
        return assert_is(
            LabelOsc(k, symbol, int(span), variant, norm), t
        )
    elif sig == "wm":
        if len(split) == 5:
            symbol, _, span, threshold, rest = split
            reen = None
        elif len(split) == 6:
            symbol, _, span, threshold, reen, rest = split
        else:
            raise ValueError(k)
        split = rest.split("-")
        if len(split) == 1:
            (variant,) = split
        elif len(split) == 2:
            variant, norm = split
        else:
            raise ValueError(k)
        return assert_is(
            LabelWm(
                k,
                symbol,
                int(span),
                parse_negatable(threshold),
                (
                    reen
                    if reen is None
                    else -1 * parse_negatable(threshold)
                    if reen == "re"
                    else parse_negatable(reen)
                ),
                variant,
                norm,
            ),
            t,
        )
    else:
        raise ValueError(k)


# V = TypeVar("V")
# R = TypeVar("R")
# P = ParamSpec("P")

# def map_with(
#     f: Callable[[Concatenate[V, P], R]],
#     vs: Iterable[V],
#     *args: P.args,
#     **kwargs: P.kwargs,
# ) -> Iterable[R]:
#     f_with_kwargs = functools.partial(f, **kwargs)
#     f_with_args = lambda v: f_with_kwargs(v, *args)
#     return map(f_with_args, vs)


@dataclass(frozen=True)
class DailyPerfVariants(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    n: int = 2
    width: float = 0.1

    def render(self) -> DailyPerfVariants:

        df = self.df
        if not len(df):
            return self

        exp = self.explorer

        df = df.pivot(
            on=exp.label_key,
            index=["date"],
            values=exp.value_key,
        )

        labels: list[LabelOsc | LabelWm | LabelConst] = (
            list(
                map(
                    functools.partial(
                        parse_label,
                        t=LabelOsc | LabelWm | LabelConst,
                    ),
                    df.drop("date").schema.keys(),
                )
            )
        )

        chart_specs = {
            "_".join(map(str, spec)): spec
            for spec in [
                (f"violin_{metric}", field, sample, window)
                for field in [
                    "span",
                ]
                for metric in [
                    "sharpe",
                    "calmar",
                ]
                for sample in [
                    "M",
                    "Y",
                    # q? y?
                ]
                for window in [
                    "M",
                    "Y",
                    # M?
                ]
                if WINDOWS[window] >= WINDOWS[sample]
            ]
            + [
                # other chart type
            ]
        }

        charts = set(
            st.segmented_control(
                "charts",
                list(chart_specs.keys()),
                selection_mode="multi",
                key=f"{self.key}_chart_ctrl",
            )
        )

        for chart_key, (
            chart,
            field,
            sample,
            window,
        ) in chart_specs.items():
            if chart_key not in charts:
                continue

            if chart.startswith("violin"):
                metric = chart.split("_")[1]
                render_metric_violin(
                    df,
                    f"{self.key}_{chart_key}",
                    field,
                    metric,
                    labels,
                    sample,
                    window,
                )

        return self


EXPLORER_COMPONENTS["perf_variants"] = DailyPerfVariants

__all__ = ["DailyPerfVariants"]
