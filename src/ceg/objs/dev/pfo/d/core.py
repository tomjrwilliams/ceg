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

from frozendict import frozendict

import numpy as np
import polars as pl

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

from ...calc import *


#  ------------------


def shift_col(
    col: str,
    shift: int | None = None,
    fill: float | None = None,
):
    if shift is None:
        return pl.col(col)
    return (
        pl.col(col).forward_fill().shift(1, fill_value=fill)
    )


def combine_backtest_dfs(
    df_rx: pl.DataFrame,
    df_scale: pl.DataFrame | None,
    df_signal: pl.DataFrame | None = None,
    df_rebalance: pl.DataFrame | None = None,
    df_calendar: pl.DataFrame | None = None,
    df_weight: pl.DataFrame | None = None,
    df_cap: pl.DataFrame | None = None,
    df_fx: pl.DataFrame | None = None,
    df_px: pl.DataFrame | None = None,
    df_gross: pl.DataFrame | None = None,
    cap: float | None = None,
    **shifts: dict,
):
    dfs = [df_rx]
    if df_scale is None:
        df_scale = df_rx.select(
            "date", pl.lit(1.0).alias("scale")
        )
    dfs.append(
        df_scale.with_columns(
            scale=shift_col(
                "scale",
                shifts.get("scale", {}).get("lag", 1),
                fill=None,
            )
        )
    )
    if df_signal is not None:
        dfs.append(
            df_signal.with_columns(
                signal=shift_col(
                    "signal",
                    shifts.get("signal", {}).get("lag", 1),
                    fill=0,
                )
            )
        )
    if df_rebalance is not None:
        dfs.append(df_rebalance)
    if df_calendar is not None:
        dfs.append(df_calendar)
    if df_weight is not None:
        dfs.append(
            df_weight.with_columns(
                weight=shift_col(
                    "weight",
                    shifts.get("weight", {}).get("lag", 1),
                    fill=None,
                ).alias("weight")
            )
        )
    if df_fx is not None:
        dfs.append(df_fx)
    if df_px is not None:
        # TODO: else 1.?
        dfs.append(df_px)
    if df_gross is not None:
        # TODO: else 1.?
        dfs.append(df_gross)
    if df_cap is not None:
        df_cap = df_cap.with_columns(
            (cap / pl.col("cap").fill_null(1.0))
            .alias("cap")
            .fill_nan(None)
        )
        dfs.append(
            df_cap.with_columns(
                cap=shift_col(
                    "cap",
                    shifts.get("cap", {}).get("lag", 1),
                    fill=0,
                )
            )
        )

    df = apis.pl.join_dfs(dfs, align=True, on="date").sort(
        by="date"
    )

    if df_weight is not None:
        if "signal" in df.schema.keys():
            df = df.with_columns(
                (
                    pl.col("signal").forward_fill()
                    * pl.col("weight").forward_fill()
                ).alias("signal")
            )
        else:
            df = df.rename(
                {"weight": "signal"}
            ).with_columns(pl.col("signal").forward_fill())

    return df


def run_backtest(
    df: pl.DataFrame,
    signal: str | None = None,
    rebal: str | None = None,
    calendar: str | None = None,
    fx: str | None = None,
    cap: str | None = None,
    unit: float = 1.0,
):
    df = apis.perf.with_backtest(
        df.with_columns(
            pl.col("rx").alias("rx_asset")
        ).sort(by="date"),
        ("pos", "rx"),
        cum=False,
        rx="rx",
        scale="scale",
        signal=signal,
        rebal=rebal,
        calendar=calendar,
        fx=fx,
        cap=cap,
        unit=unit,
    )
    df = df.with_columns(
        pl.when(pl.col("gross").is_null())
        .then(None)
        .otherwise(
            pl.col("gross")
            .forward_fill()
            .mul(pl.col("pos").forward_fill().abs())
        )
        .alias("gross")
    ).with_columns(
        pl.when(pl.col("px").is_null())
        .then(None)
        .otherwise(
            pl.col("px")
            .forward_fill()
            .mul(pl.col("pos").forward_fill())
        )
        .alias("px")
    )

    # assert len(df) == len(df_rx), dict(
    #     df=df, rx=df_rx
    # )

    return df


import numpy as np


def mutate_df(
    df: pl.DataFrame | None,
    col: str,
    rel_mu: float | None = None,
    rel_sd: float | None = None,
    abs_mu: float | None = None,
    abs_sd: float | None = None,
    lag: int | None = None,
    lower: float | None = None,
    upper: float | None = None,
    sdevs: int | None = None,
    sdev_width: float | None = None,
    sdev: int | None = None,
    quantiles: int | None = None,
    quantile: int | None = None,
):
    # NOTE: ignore lag, dealt with in run_backtest?

    if df is None:
        return df

    e = pl.col(col)
    e_sq = e.pow(2)

    # sd = e.rolling_std(len(df), min_samples=2)
    sd = e_sq.rolling_mean(len(df), min_samples=2).sqrt()
    mu = e.rolling_mean(len(df), min_samples=2)

    if rel_sd is not None:
        raise ValueError(rel_sd)
        if rel_mu is not None:
            err = np.random.normal(
                mu * rel_mu, scale=sd * rel_sd, size=len(df)
            )
        else:
            err = np.random.normal(
                0.0, scale=sd * rel_sd, size=len(df)
            )

        df = df.with_columns(
            (e + pl.Series(col, err)).alias(col)
        )

    if abs_sd is not None:
        if abs_mu is not None:
            err = np.random.normal(
                abs_mu, scale=abs_sd, size=len(df)
            )
        else:
            err = np.random.normal(
                0, scale=abs_sd, size=len(df)
            )
        df = df.with_columns(
            (e + pl.Series(col, err)).alias(col)
        )

    if lower is not None:
        df = df.with_columns(
            apis.pl.mask_col(col, lambda e: e >= lower)
        )

    if upper is not None:
        df = df.with_columns(
            apis.pl.mask_col(col, lambda e: e <= upper)
        )

    l_v = r_v = None

    if quantiles is not None or quantile is not None:
        assert quantile is not None, quantile
        assert quantiles is not None, quantiles

        w = 1 / quantiles
        l = None if quantile == 0 else quantile * w
        r = (
            None
            if quantile == quantiles - 1
            else (quantile + 1) * w
        )

        l_v = (
            None
            if l is None
            else e.rolling_quantile(
                l, window_size=len(df), min_samples=10
            )
        )
        r_v = (
            None
            if r is None
            else e.rolling_quantile(
                r, window_size=len(df), min_samples=10
            )
        )

        assert l_v is not None or r_v is not None, (
            quantiles,
            quantile,
        )

    if sdevs is not None or sdev is not None:
        assert sdev is not None, sdev
        assert sdevs is not None, sdevs

        # TODO: centre optional

        l, r = apis.stats.sdev_increments(
            sdev_width, sdevs, mu=0
        )[sdev]
        l_v = None if l is None else l * sd
        r_v = None if r is None else r * sd

        assert l is not None or r is not None, (
            sdevs,
            sdev_width,
        )

    if l_v is not None or r_v is not None:
        df = df.with_columns(
            apis.pl.mask_col(
                col,
                lambda e: (
                    (
                        pl.lit(True)
                        if l_v is None
                        else e > l_v
                    )
                    & (
                        pl.lit(True)
                        if r_v is None
                        else e < r_v
                    )
                ),
            )
        )

    return df


def rebalance_df(
    rebalance: int | str | Daily | None,
    df_index: pl.DataFrame | None = None,
    objects: tuple[Object | str, ...] | None = None,
    alias: str = "rebalance",
):
    return (
        None
        if rebalance is None
        else apis.t.not_none(df_index).select(
            "date",
            apis.cal.rebalance_mask(rebalance).alias(alias),
        )
        if isinstance(rebalance, str)
        else daily_pl_values("df", rebalance, alias=alias)
        if isinstance(rebalance, Daily)
        else daily_pl_values(
            "df", rebalance, objs=objects, alias=alias
        )
    )


__all__ = [
    "combine_backtest_dfs",
    "run_backtest",
    "rebalance_df",
]
