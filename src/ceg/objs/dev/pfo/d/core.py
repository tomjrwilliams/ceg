
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

from ..bt import *

#  ------------------


def run_backtest(
    df_rx: pl.DataFrame,
    df_scale: pl.DataFrame,
    df_signal: pl.DataFrame | None = None,
    df_rebalance: pl.DataFrame | None = None,
    df_calendar: pl.DataFrame | None = None,
    df_weight: pl.DataFrame | None = None,
    df_cap: pl.DataFrame | None = None,
    df_fx: pl.DataFrame | None = None,
    # TODO: df_cap
    # TODO: take shifts as args
    unit: float = 1.,
):
    df = df_rx.join(
        df_scale,
        on="date",
        how="full",
        coalesce=True,
    ).sort(by="date").with_columns(
        scale=(
            pl.col("scale").forward_fill()
            .shift(1, fill_value=None)
        )
    )
    if df_signal is not None:
        df = df.join(
            df_signal,
            on="date",
            coalesce=True,
            how="full"
        ).sort(by="date").with_columns(
            signal=(
                pl.col("signal").forward_fill()
                .shift(1, fill_value=1)
            )
        )
    if df_rebalance is not None:
        df = df.join(
            df_rebalance,
            on="date",
            coalesce=True,
            how="full"
        )
    if df_calendar is not None:
        df = df.join(
            df_calendar,
            on="date",
            coalesce=True,
            how="full"
        )
    if df_weight is not None:
        df = df.join(
            df_weight,
            on="date",
            coalesce=True,
            how="left"
        ).sort(by="date").with_columns(
            weight=(
                pl.col("weight").forward_fill()
                .shift(1, fill_value=None)
            )
        ).with_columns(
            (pl.col("scale") * pl.col("weight"))
            .alias("scale")
        )
    if df_fx is not None:
        df = df.join(
            df_fx,
            on="date",
            coalesce=True,
            how="full"
        )
    if df_cap is not None:
        df = df.join(
            df_cap,
            on="date",
            coalesce=True,
            how="full"
        ).sort(by="date").with_columns(
            cap=(
                pl.col("cap").forward_fill()
                .shift(1, fill_value=0)
            )
        )
        
    df = apis.perf.with_backtest(
        df.sort(by="date"),
        ("pos", "rx"),
        cum=False,
        rx="rx",
        scale="scale",
        signal=None if df_signal is None else "signal",
        rebal=None if df_rebalance is None else "rebalance",
        calendar=None if df_calendar is None else "calendar",
        fx=None if df_fx is None else "fx",
        cap=None if df_cap is None else "cap",
        unit=unit
    )

    # assert len(df) == len(df_rx), dict(
    #     df=df, rx=df_rx
    # )

    return df

def get_backtest_dfs(
    bt: Backtest,
    domain: Domain,
    univ: Universe,
    with_rx: bool = True,
):

    # TODO: if any are none that shouldnt be, skip?
    if with_rx:
        assert bt.rx is not None, bt
        df_rx = daily_pl_values(
            domain,
            "df",
            bt.rx,
            alias="rx",
            objs=univ.objects
        )
    else:
        df_rx = None

    if bt.scale is None:
        df_scale = None
    else:
        scale_i, scale_v, scale_agg = bt.get_scale()
        df_scale = daily_pl_values(
            domain,
            "df",
            scale_i, 
            scale=scale_v,
            agg=scale_agg,
            alias="scale",
            objs=univ.objects
        )
    if bt.signal is None:
        df_signal = None
    else:
        sig_i, sig_v, sig_agg = bt.get_signal()
        df_signal = daily_pl_values(
            domain,
            "df",
            sig_i, 
            scale=sig_v,
            agg=sig_agg,
            alias="signal",
            objs=univ.objects
        )

    df_index = (
        df_scale if df_scale is not None
        else df_rx if df_rx is not None
        else df_signal
    )
    if isinstance(bt.rebalance, str):
        assert df_index is not None, bt

    df_rebalance = (
        None if bt.rebalance is None
        else apis.t.not_none(df_index).select(
            "date",
            apis.cal.rebalance_mask(bt.rebalance)
            .alias("rebalance")
        )
        if isinstance(bt.rebalance, str)
        else daily_pl_values(
            domain,
            "df",
            bt.rebalance,
            objs=univ.objects,
            alias="rebalance"
        )
    )
    df_calendar = (
        None if bt.calendar is None 
        else daily_pl_values(
            domain,
            "df",
            bt.calendar,
            objs=univ.objects,
            alias="calendar",
        )
    )
    df_weight = (
        None if bt.weight is None 
        else daily_pl_values(
            domain,
            "df",
            bt.weight,
            objs=univ.objects,
            alias="weight",
        )
    )
    df_cap = (
        None if bt.cap is None 
        else daily_pl_values(
            domain,
            "df",
            bt.cap,
            objs=univ.objects,
            alias="cap",
        )
    )

    df_fx = (
        None if bt.fx is None 
        else daily_pl_values(
            domain,
            "df",
            bt.fx,
            objs=univ.objects,
            alias="fx",
        )
    )

    # TODO: unit cap

    return (
        df_rx,
        df_scale,
        df_signal,
        df_rebalance,
        df_calendar,
        df_weight,
        df_cap, 
        df_fx,
    )

__all__ = [
    "get_backtest_dfs",
    "run_backtest",
]