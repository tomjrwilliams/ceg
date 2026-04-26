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
from .utils import *


def pivot_ratio(sub_df: pl.DataFrame, value_key: str):
    ratio_keys = [
        (
            f"{hl}wm_{k1}",
            f"{hl}wm_{k2}",
        )
        for hl in ["h", "l"]
        for (k1, k2) in [
            ("net", "len"),
            ("grs", "len"),
            ("net", "grs"),
        ]
        if (
            f"{hl}wm_{k1}" in sub_df.schema.keys()
            and f"{hl}wm_{k2}" in sub_df.schema.keys()
        )
    ]
    if len(ratio_keys):
        return sub_df.with_columns(
            *(
                (pl.col(k1) / pl.col(k2))
                .replace(0, None)
                .fill_nan(None)
                .alias(f"{k1}_{k2}")
                for k1, k2 in ratio_keys
            )
        ).unpivot(
            on=[f"{k1}_{k2}" for k1, k2 in ratio_keys],
            index=["date", "symbol", "span"],
            variable_name="field",
            value_name="val",
        )
    else:
        return None


@dataclass(frozen=True)
class DailyWatertrails(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    n: int = 2
    width: float = 0.1

    def render(self) -> DailyWatertrails:

        df = self.df
        if not len(df):
            return self

        exp = self.explorer
        # df = self.render_date_filter(df)

        val = pl.col(exp.value_key)

        df = df.pipe(
            with_col_split_into,
            "group",
            on="-",
            names=[
                "symbol",
                "span",
                "field",
            ],
        )

        charts = st.segmented_control(
            "charts",
            [
                exp.value_key,
                f"{exp.value_key}_rx",
                "log",
                "log_rx",
                f"{exp.value_key}_cond",
                "log_cond",
                "ratio",
            ],
            selection_mode="multi",
            key=f"{self.key}_chart_ctrl",
        )

        is_len_e = pl.col("field").is_in(
            [
                "hwm_len",
                "lwm_len",
            ]
        )
        if_len_0_else_1 = (
            pl.when(is_len_e).then(0).otherwise(1)
        )

        df_state = (
            df.pipe(
                radio_filter_df,
                [
                    "symbol",
                    "span",
                    "field",
                ],
                self.key,
            )
            .with_columns(
                pl.when(pl.col("field") == "hwm_net")
                .then(val.clip(upper_bound=0))
                .when(pl.col("field") == "lwm_net")
                .then(val.clip(lower_bound=0))
                .otherwise(pl.col(exp.value_key))
                .alias(exp.value_key)
            )
            .with_columns(
                pl.when(val < 0)
                .then(
                    val.mul(-1)
                    .add(if_len_0_else_1)
                    .log()
                    .mul(-1)
                )
                .otherwise(val.add(if_len_0_else_1).log())
                .alias("log")
            )
        )

        figs_line = {
            k + suff: go.Figure()
            for k in [exp.value_key, "log"]
            for suff in ["", "_rx"]
        }
        figs_cond = {
            k: go.Figure() for k in [exp.value_key, "log"]
        }

        fig_ratio = go.Figure()

        # span_32 = {
        #     v_key: df_state.filter(
        #         pl.col("span") == "32.0",
        #         pl.col("field").is_in([
        #             "osc_15",
        #             "osc_30",
        #         ])
        #     ).with_columns(
        #         pl.col("field")
        #         .replace({
        #             "osc_15": "osc_15_32",
        #             "osc_30": "osc_30_32",
        #         })
        #         .alias("field")
        #     ).pivot(
        #         on="field",
        #         index=[
        #             "date", "symbol",
        #         ],
        #         values=v_key,
        #     )
        #     for v_key in [exp.value_key, "log"]
        # }

        for (label,), sub_df in sorted(
            df_state.partition_by(
                exp.label_key, as_dict=True
            ).items(),
            key=lambda k_df: k_df[0],
        ):
            sub_df = sub_df.sort(by="date")

            pivots = {
                v_key: sub_df.pivot(
                    on="field",
                    index=[
                        "date",
                        "symbol",
                        "span",
                    ],
                    values=v_key,
                )
                # .join(
                #     span_32[v_key],
                #     on=["date", "symbol"],
                #     how="left",
                #     coalesce=True,
                # )
                for v_key in [exp.value_key, "log"]
            }

            for v_key, field_df in pivots.items():
                if f"{v_key}_cond" not in charts:
                    continue
                field_df = pivots[v_key]
                if v_key == "log":
                    f_quantile = lambda e, i, k, max_len: (
                        e.rolling_quantile(
                            (
                                i
                                if k != "hwm_net"
                                else 10 - i
                            )
                            / 10,
                            window_size=max_len,
                            min_samples=100,
                        )
                    )
                else:
                    f_quantile = lambda e, i, k, max_len: (
                        e.rolling_max(
                            window_size=max_len,
                            min_samples=100,
                        )
                        * (i / 10)
                        if k != "hwm_net"
                        else e.rolling_min(
                            window_size=max_len,
                            min_samples=100,
                        )
                        * (i / 10)
                    )

                for hl in ["h", "l"]:
                    if (
                        f"{hl}wm_len"
                        not in field_df.schema.keys()
                    ):
                        continue

                    cond_df_pivot = quantile_condition_by(
                        field_df,
                        start=pl.col(f"{hl}wm_len").eq(
                            1 if v_key != "log" else 0
                        ),
                        keys=[
                            f"{hl}wm_{k}"
                            for k in ["len", "net", "grs"]
                        ],
                        key_name="field",
                        quantile=f_quantile,
                        comp=(
                            lambda e, q, k: (
                                e.gt(q)
                                if k != "hwm_net"
                                else e.lt(q)
                            )
                        ),
                    )
                    for t in (
                        px.scatter(
                            cond_df_pivot,
                            x=f"v_last",
                            y="qile",
                            color="field",
                        )
                        .update_traces(
                            legendgroup=label,
                            legendgrouptitle_text=label,
                        )
                        .data
                    ):
                        figs_cond[v_key].add_trace(t)

            for v_key, pivot_df in pivots.items():
                if (
                    f"{v_key}_rx" not in charts
                    and v_key not in charts
                ):
                    continue
                ks = [
                    k
                    for k in ["net", "grs", "len"]
                    if (
                        f"hwm_{k}" in pivot_df.schema
                        and f"lwm_{k}" in pivot_df.schema
                    )
                ]
                if len(ks):
                    pivot_df = (
                        pivot_df.with_columns(
                            pl.when(
                                pl.col(f"hwm_{k}")
                                .fill_nan(None)
                                .forward_fill()
                                .abs()
                                .pipe(
                                    op,
                                    pl.col(f"lwm_{k}")
                                    .fill_nan(None)
                                    .forward_fill(),
                                )
                            )
                            .then(
                                pl.col(f"hwm_{k}")
                                .fill_nan(None)
                                .forward_fill()
                                .abs()
                            )
                            .otherwise(
                                pl.col(f"lwm_{k}")
                                .fill_nan(None)
                                .forward_fill()
                                .abs()
                            )
                            .alias(f"{pref}_{k}")
                            for k in ks
                            for pref, op in {
                                "max": lambda hwm, lwm: (
                                    hwm.gt(lwm)
                                ),
                                "min": lambda hwm, lwm: (
                                    hwm.lt(lwm)
                                ),
                            }.items()
                        )
                        .with_columns(
                            *(
                                pl.col(f"min_{k}")
                                .pow(2)
                                .rolling_mean(
                                    len(pivot_df),
                                    min_samples=2,
                                )
                                .sqrt()
                                .alias(f"rms_{k}")
                                for k in ks
                            )
                        )
                        .with_columns(
                            *(
                                pl.col(f"min_{k}")
                                .abs()
                                .rolling_mean(
                                    len(pivot_df),
                                    min_samples=2,
                                )
                                .alias(f"abs_{k}")
                                for k in ks
                            )
                        )
                        .pipe(
                            with_osc_sig_rx,
                        )
                        .pipe(
                            with_wm_sig_rx,
                            [
                                # "net",
                                "len",
                                # "grs"
                            ],
                            [
                                # "abs",
                                "rms"
                            ],
                            [
                                # 1.5,
                                2.0,
                                # 2.5,
                            ],
                            [False, True],
                            [
                                False,
                                # True
                            ],
                            [
                                "thold",
                                # "gt"
                            ],
                        )
                    )
                    pivot_df = pivot_df.unpivot(
                        on=[
                            k
                            for k in [
                                f"{pref}_{k}"
                                for k in ks
                                for pref in [
                                    "max",
                                    "min",
                                    "rms",
                                    "abs",
                                ]
                                + [
                                    f"{sig}{pref}_{sc}_{m}{re}{lo}{suff}"
                                    for pref in ["", "_rx"]
                                    for sc in ["abs", "rms"]
                                    for m in [1.5, 2.0, 2.5]
                                    for re in ["", "_re"]
                                    for lo in ["", "_ls"]
                                    for sig in [
                                        "thold",
                                        "gt",
                                    ]
                                    for suff in [
                                        "",
                                        # "_w_so"
                                    ]
                                ]
                            ]
                            + [
                                f"sig_osc_{sp}"
                                for sp in [
                                    # "15",
                                    "30"
                                ]
                            ]
                            + [
                                f"rx_osc_{sp}_{var}"
                                for sp in [
                                    # "15",
                                    "30",
                                ]
                                for var in [
                                    # "ls",
                                    # "lo",
                                    # "so",
                                    # "ls_b",
                                    "lo_b",
                                    # "lo_b_w_so",
                                    "so_b",
                                ]
                            ]
                            if k in pivot_df.schema
                        ],
                        index=[
                            "symbol",
                            "span",
                            exp.date_key,
                        ],
                        variable_name="field",
                        value_name=v_key,
                    ).with_columns(
                        pl.concat_str(
                            pl.col("symbol"),
                            pl.col("span"),
                            pl.col("field"),
                            separator="-",
                        ).alias("group")
                    )
                    pivot_df = sub_df.select(
                        exp.date_key, v_key, "group"
                    ).vstack(
                        pivot_df.select(
                            exp.date_key, v_key, "group"
                        )
                    )
                else:
                    pivot_df = sub_df

                if v_key in charts:
                    pivot_no_rx = pivot_df.filter(
                        pl.col("group")
                        .str.contains("rx", literal=True)
                        .not_()
                    )
                    for k in ["net", "grs", "len", "osc"]:
                        for t in (
                            pivot_no_rx.filter(
                                pl.col(
                                    "group"
                                ).str.contains(
                                    k, literal=True
                                )
                            )
                            .pipe(
                                px.line,
                                x=exp.date_key,
                                y=v_key,
                                color="group",
                            )
                            .update_traces(
                                legendgroup=f"{label}_{k}",
                                legendgrouptitle_text=f"{label}_{k}",
                            )
                            .data
                        ):
                            figs_line[v_key].add_trace(t)

                if f"{v_key}_rx" in charts:
                    pivot_rx = pivot_df.filter(
                        pl.col("group").str.contains(
                            "rx", literal=True
                        )
                    )
                    if len(pivot_rx):
                        for t in (
                            px.line(
                                pivot_rx.select(
                                    exp.date_key,
                                    "group",
                                    pl.col(v_key)
                                    .cum_sum()
                                    .over("group"),
                                ),
                                x=exp.date_key,
                                y=v_key,
                                color="group",
                            )
                            .update_traces(
                                legendgroup=label,
                                legendgrouptitle_text=label,
                            )
                            .data
                        ):
                            figs_line[
                                v_key + "_rx"
                            ].add_trace(t)

            if "ratio" not in charts:
                sub_df_ratio = pivot_ratio(
                    pivots[exp.value_key], exp.value_key
                )
                if sub_df_ratio is not None:
                    for t in (
                        px.line(
                            sub_df_ratio,
                            x=exp.date_key,
                            y="val",
                            color="field",
                        )
                        .update_traces(
                            legendgroup=label,
                            legendgrouptitle_text=label,
                        )
                        .data
                    ):
                        fig_ratio.add_trace(t)

        for v_key, fig in figs_line.items():
            if v_key not in charts:
                continue
            st.plotly_chart(
                fig.update_layout(
                    legend=dict(groupclick="toggleitem")
                ),
                key=f"{self.key}_fig_line_{v_key}",
            )

        for v_key, fig in figs_cond.items():
            if f"{v_key}_cond" not in charts:
                continue
            st.plotly_chart(
                fig.update_layout(
                    legend=dict(groupclick="toggleitem")
                ),
                key=f"{self.key}_fig_cond_{v_key}",
            )

        if "ratio" in charts:
            st.plotly_chart(
                fig_ratio.update_layout(
                    legend=dict(groupclick="toggleitem")
                ),
                key=f"{self.key}_fig_ratio",
            )

        return self


def with_wm_sig_rx(
    df: pl.DataFrame,
    ks: list[str],
    scale: list[str],
    muls: list[float] = [2.0],
    re_entry: list[bool] = [False],
    ls: list[bool] = [False],
    sigs: list[str] = ["thold"],
):
    for k in ks:
        if f"min_{k}" not in df.schema:
            continue
        for sc in scale:
            if f"{sc}_{k}" not in df.schema:
                continue
            for mul in muls:
                for re_en in re_entry:
                    # if re_en and k != "net":
                    #     continue
                    for lo in ls:
                        for sig in sigs:
                            if re_en and sig == "gt":
                                continue
                            df = df.pipe(
                                with_wm_sig,
                                k,
                                sc,
                                mul,
                                re_en,
                                lo,
                                sig,
                            ).pipe(
                                with_wm_rx,
                                k,
                                sc,
                                mul,
                                re_en,
                                lo,
                                sig,
                            )
    return df


def with_osc_sig_rx(
    df: pl.DataFrame,
    vol_span: int = 48,
    sig_span: int | None = None,
    activation=lambda e: e.tanh(),
):
    e_rx = pl.col("rx")

    e_rx_rms = (
        e_rx.pow(2)
        .ewm_mean(span=vol_span, min_samples=2)
        .sqrt()
        .fill_nan(None)
    )

    e_osc_15 = pl.col("osc_15") / (e_rx_rms)
    e_osc_30 = pl.col("osc_30") / (e_rx_rms)

    # e_osc_15_32 = pl.col("osc_15_32") / (e_rx_rms)
    # e_osc_30_32 = pl.col("osc_30_32") / (e_rx_rms)

    f_sig_rms = lambda e: (
        e.ewm_mean(span=sig_span, min_samples=2)
        if sig_span is not None
        else e.rolling_mean(len(df), min_samples=2)
    )

    e_osc_15_rms = (
        e_osc_15.pow(2)
        .pipe(f_sig_rms)
        .sqrt()
        .fill_nan(None)
    )
    e_osc_30_rms = (
        e_osc_30.pow(2)
        .pipe(f_sig_rms)
        .sqrt()
        .fill_nan(None)
    )
    # e_osc_15_32_rms = (
    #     e_osc_15_32.pow(2)
    #     .pipe(f_sig_rms)
    #     .sqrt()
    #     .fill_nan(None)
    # )
    # e_osc_30_32_rms = (
    #     e_osc_30_32.pow(2)
    #     .pipe(f_sig_rms)
    #     .sqrt()
    #     .fill_nan(None)
    # )

    f_rx = lambda k_sig, k_rx: (
        pl.col("rx")
        .mul(
            pl.col(k_sig)
            .shift(1)
            .fill_nan(None)
            .forward_fill()
        )
        .alias(k_rx)
    )
    # f_rx_mod = lambda k_sig, k_rx: (
    #     pl.col("rx")
    #     .mul(
    #         pl.when(
    #             pl.col("sig_osc_30_32_so_b")
    #             < 0
    #         )
    #         .then(0)
    #         .otherwise(pl.col(k_sig))
    #         .shift(1)
    #         .fill_nan(None)
    #         .forward_fill()
    #     )
    #     .alias(k_rx)
    # )

    f_lt = lambda s1, s2: (
        pl.when(
            pl.col(f"sig_osc_{s1}").lt(
                pl.col(f"sig_osc_{s2}")
            )
        )
        .then(pl.col(f"sig_osc_{s1}"))
        .otherwise(pl.col(f"sig_osc_{s2}"))
    )
    f_gt = lambda s1, s2: (
        pl.when(
            pl.col(f"sig_osc_{s1}").gt(
                pl.col(f"sig_osc_{s2}")
            )
        )
        .then(pl.col(f"sig_osc_{s1}"))
        .otherwise(pl.col(f"sig_osc_{s2}"))
    )

    osc_var = {
        f"sig_osc_{sp}_{var}": f(pl.col(f"sig_osc_{sp}"))
        for sp in [
            # "15",
            "30",
            # "30_32"
        ]
        for var, f in {
            "ls": lambda e: e,
            "lo": lambda e: e.clip(lower_bound=0),
            "so": lambda e: e.clip(upper_bound=0),
            "ls_b": lambda e: e.pipe(to_binary),
            "lo_b": lambda e: e.pipe(to_binary).clip(
                lower_bound=0
            ),
            "so_b": lambda e: e.pipe(to_binary).clip(
                upper_bound=0
            ),
        }.items()
    }

    return (
        df.with_columns(
            # (e_osc_15 / e_osc_15_rms)
            # .tanh()
            # .fill_nan(None)
            # .alias("sig_osc_15"),
            (e_osc_30 / e_osc_30_rms)
            .tanh()
            .fill_nan(None)
            .alias("sig_osc_30"),
            # (e_osc_30_32 / e_osc_30_32_rms)
            # .tanh()
            # .fill_nan(None)
            # .alias("sig_osc_30_32"),
        )
        .with_columns(
            e.alias(k) for k, e in osc_var.items()
        )
        .with_columns(
            f_rx(
                f"sig_osc_{sp}_{var}",
                f"rx_osc_{sp}_{var}",
            )
            for sp in [
                # "15",
                "30"
            ]
            for var in [
                "ls",
                "lo",
                "so",
                "ls_b",
                "lo_b",
                "so_b",
            ]
            # ).with_columns(
            #     f_rx_mod(
            #         f"sig_osc_{sp}_{var}",
            #         f"rx_osc_{sp}_{var}_w_so",
            #     )
            #     for sp in [
            #         "30"
            #     ]
            #     for var in ["lo_b"]
        )
    )


# TODO: if 32 span osc so is active, 0, else signal (for all lo osc and hwm)


def to_binary(e: pl.Expr):
    return (
        pl.when(e == 0)
        .then(0)
        .when(e > 0)
        .then(1.0)
        .when(e < 0)
        .then(-1)
    )


def with_wm_rx(
    df: pl.DataFrame,
    k: str,
    scale: str = "abs",
    mul: float = 2.0,
    re_entry: bool = True,
    ls: bool = False,
    sig: str = "thold",
):
    re = "_re" if re_entry else ""
    lo = "_ls" if ls else ""
    if "rx" not in df.schema:
        return df
    k_sig = f"{sig}_{scale}_{mul}{re}{lo}"
    k_rx = f"{sig}_rx_{scale}_{mul}{re}{lo}"
    # for suff in ["", "_w_so"]:
    for suff in [""]:
        df = df.with_columns(
            pl.col("rx")
            .mul(
                pl.col(k_sig + suff + f"_{k}")
                .shift(1)
                .fill_nan(None)
                .forward_fill()
            )
            .alias(k_rx + suff + f"_{k}")
        )
    return df


def with_wm_sig(
    df: pl.DataFrame,
    k: str,
    scale: str = "abs",
    mul: float = 2.0,
    re_entry: bool = True,
    ls: bool = False,
    sig: str = "thold",
):
    fill = lambda s: (
        s.fill_nan(None).forward_fill().to_numpy()
    )

    nd_min = fill(df.get_column(f"min_{k}"))
    nd_hwm = fill(df.get_column(f"hwm_{k}"))
    nd_lwm = fill(df.get_column(f"lwm_{k}"))
    nd_rms = fill(df.get_column(f"{scale}_{k}"))

    nd_pos_thresh = np.zeros(nd_min.shape)
    nd_pos_gt = np.zeros(nd_min.shape)

    k_mul = -1 if k == "net" else 1.0

    in_dd = False

    pos_neg = -1 if ls else 0
    pos_thresh = 1
    pos_gt = 1

    hwm_entry = 0

    for t in range(len(df)):
        v_min = nd_min[t]
        v_hwm = nd_hwm[t]
        v_lwm = nd_lwm[t]
        v_rms = nd_rms[t]

        pos_gt = 1 if abs(v_lwm) > abs(v_hwm) else pos_neg

        if not in_dd and (k_mul * v_hwm) > (mul * v_rms):
            in_dd = True
            pos_thresh = pos_neg
        elif not in_dd:
            pass
        elif in_dd and (k_mul * v_hwm) < v_rms:
            in_dd = False
            pos_thresh = 1
        elif (
            re_entry
            and in_dd
            and pos_thresh == pos_neg
            and v_lwm > mul * v_rms
        ):
            pos_thresh = 1
            hwm_entry = k_mul * v_hwm
        elif (
            re_entry
            and in_dd
            and pos_thresh == 1
            and ((v_hwm * k_mul) > (hwm_entry + v_rms))
        ):
            pos_thresh = pos_neg
        else:
            pass

        nd_pos_thresh[t] = pos_thresh
        nd_pos_gt[t] = pos_gt

    re = "_re" if re_entry else ""
    lo = "_ls" if ls else ""

    nd_res = assert_is(
        nd_pos_thresh
        if sig == "thold"
        else nd_pos_gt
        if sig == "gt"
        else None
    )

    res_k = f"{sig}_{scale}_{mul}{re}{lo}"
    return df.with_columns(
        pl.Series(res_k + f"_{k}", nd_res)
        # ).with_columns(
        #     pl.when(pl.col("sig_osc_30_32_so_b") < 0)
        #     .then(0)
        #     .otherwise(pl.col(res_k+ f"_{k}"))
        #     .alias(f"{res_k}_w_so" + f"_{k}")
    )


EXPLORER_COMPONENTS["watertrails"] = DailyWatertrails

__all__ = ["DailyWatertrails"]
