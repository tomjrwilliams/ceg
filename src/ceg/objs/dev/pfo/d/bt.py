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
from .....apis import assert_is, t, kv, logs
from .....engine import *

from ...calc import *
from .....apis.calcs.weights import pfo_weights_df

from .core import *

logger, log = logs.logger(
    __file__,
    streams=dict(formatter=True),
)


#  ------------------


def combine_mv_dfs(
    df_uv: pl.DataFrame | None,
    df_mv: pl.DataFrame | None,
    k: str,
):
    if df_uv is None:
        return df_mv
    elif df_mv is None:
        return df_uv
    return (
        df_uv.join(
            df_mv.rename({k: VALUE}),
            on="date",
            how="full",
            coalesce=True,
        )
        .sort(by="date")
        .select(
            "date", (pl.col(VALUE) * pl.col(k)).alias(k)
        )
    )


def combine_mv_scale(
    df_uv: pl.DataFrame | None,
    df_mv: pl.DataFrame | None,
):
    return combine_mv_dfs(df_uv, df_mv, "scale")


def combine_mv_signal(
    df_uv: pl.DataFrame | None,
    df_mv: pl.DataFrame | None,
):
    return combine_mv_dfs(df_uv, df_mv, "signal")



def rx_price_types(
    rx: Daily
):
    from ...bars import FutDailyRx, EqDailyRx, FxDailyRx
    
    if isinstance(
        rx,
        (
            FutDailyRx,
            EqDailyRx,
            FxDailyRx,
        ),
    ):
        return [type(rx)]
    elif isinstance(rx, DailyBtRx):
        return rx.get_parent().price_types()
    # elif isinstance(rx, DailyPfolioRx):
    #     raise ValueError(rx)
    else:
        raise ValueError(rx)


def rx_price_symbols(
    rx: Daily
):
    from ...bars import FutDailyRx, EqDailyRx, FxDailyRx


    if isinstance(
        rx,
        (
            FutDailyRx,
            EqDailyRx,
            FxDailyRx,
        ),
    ):
        px = rx.get_parent()
        return [px.params.symbol]
    elif isinstance(rx, DailyBtRx):
        return rx.get_parent().price_symbols()
    # elif isinstance(rx, DailyPfolioRx):
    #     raise ValueError(rx)
    else:
        raise ValueError(rx)

def recurse_bt_pos(
    bt_pos: DailyBtPos,
    obj_rx: Daily,
    key: str | None = None,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):
    df_pos = daily_pl_values(
        "df",
        bt_pos,
        alias=VALUE,
    )
    if acc is not None:
        acc = acc.join(
            df_pos,
            on="date",
            how="full",
            coalesce=True,
        ).with_columns(
            pl.col(VALUE)
            .mul(
                1
                if parent is None
                else pl.col(parent)
                .fill_nan(None)
                .fill_null(1)
            )
            .alias(VALUE)
        )
    else:
        acc = df_pos

    if key is not None:
        
        key = f"{key}={obj_rx.id}"

        if acc is None:
            return df_pos.rename({VALUE: key})
        return acc.with_columns(
            pl.col(VALUE)
            .add(
                (
                    0
                    if key not in acc.columns
                    else pl.col(key)
                    .fill_nan(None)
                    .fill_null(0)
                )
            )
            .alias(key)
        ).drop(VALUE)

    acc = acc.rename({VALUE: bt_pos.id})


    bt_parent = get_parent(obj_rx)
    assert isinstance(
        bt_parent,
        (DailyStrategy, DailyBasket)
    ), obj_rx

    return acc_bt_pos(
        bt_parent,
        parent=bt_pos.id,
        acc=acc,
    ).drop(bt_pos.id)


def acc_bt_pos_uvar(
    bt: DailyStrategy,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):
    bt_pos = assert_is(bt.bt_pos)
    
    return recurse_bt_pos(
        bt_pos, assert_is(bt.rx), key = bt.params.key, parent=parent, acc=acc
    )


def acc_bt_pos_mvar(
    bt: DailyBasket,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
):
    for i, bt_pos in enumerate(
        assert_is(bt.bt_pos)[1],
    ):
        uv = assert_is(bt.assets)[i]
        acc = recurse_bt_pos(
            bt_pos,
            assert_is(uv.rx), 
            key = uv.params.key,
            parent=parent,
            acc=acc
        )

    assert isinstance(acc, pl.DataFrame), bt
    return acc


def acc_bt_pos(
    bt: DailyStrategy | DailyBasket,
    parent: str | None = None,
    acc: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if isinstance(bt, DailyStrategy):
        return acc_bt_pos_uvar(bt, parent, acc)
    elif isinstance(bt, DailyBasket):
        return acc_bt_pos_mvar(bt, parent, acc)
    else:
        raise ValueError(bt)


@dataclass(frozen=True)
class DailyPfolioSeries:
    key: str
    rx_id: str
    asset: t.pl.DataFrame  # rx
    rx: t.pl.DataFrame  # scaled
    pos: t.pl.DataFrame


@dataclass(frozen=True)
class DailyPfolioData:
    bt: DailyStrategy | DailyBasket
    start: dt.date
    end: dt.date
    series: list[DailyPfolioSeries]

# TODO: eventually need to be able to pass daily_pfolio.rx[k]
# as an asset, so below will need to handle both paths

def pfolio_keys(bt: DailyStrategy | DailyBasket):
    if isinstance(bt, DailyStrategy):
        if bt.params.key is not None:
            return set([bt.params.key])
        assert isinstance(bt.rx, (
            DailyStrategy, DailyBasket
        ))
        return pfolio_keys(bt.rx)
    keys = set()
    for uv in assert_is(bt.assets):
        keys = keys | pfolio_keys(uv)
    return keys

def get_daily_pfolio_data(
    bt: DailyStrategy | DailyBasket
):

    df_pos = acc_bt_pos(bt).sort(by="date")

    (first, last) = df_pos.pipe(
        apis.pl.select_min_max, "date", dt.date
    )
    df_pos = pl.select(
        pl.date_range(first, last).alias("date")
    ).join(df_pos, on="date", how="left", coalesce=True)

    res = []

    for k_rx in df_pos.columns:
        if k_rx == "date":
            continue

        df_pos = df_pos.with_columns(
            pl.col(k_rx).alias("pos")
        )

        # log(get_pfo_data=k_rx)

        k, rx_path = k_rx.split("=")
        obj_rx = context.get().get(rx_path, Daily)

        df_asset = daily_pl_df(obj_rx, alias="rx")

        df_rx = df_asset.join(
            df_pos.select("date", "pos"),
            on="date",
            how="full",
            coalesce=True,
        ).sort(by="date")

        v_pos, v_pos_null = apis.pl.to_numpy_with_null(
            df_rx.get_column("pos")
        )
        v_rx, v_rx_null = apis.pl.to_numpy_with_null(
            df_rx.get_column("rx")
        )

        df_rx = df_rx.with_columns(
            pl.Series(
                apis.perf.backtest_rx_pos(
                    v_rx,
                    v_pos,
                    v_rx_null,
                    v_pos_null,
                )
            )
            .alias("rx")
            .fill_nan(None)
        ).drop("pos")

        # TODO: check sum!

        res.append(
            DailyPfolioSeries(
                k,
                rx_path,
                # k_rx = {k}={rx_path}
                t.pl.DataFrame.cast(df_asset),
                t.pl.DataFrame.cast(df_rx),
                t.pl.DataFrame.cast(df_pos),
            )
        )

    return DailyPfolioData(bt, first, last, res)


@dataclass(frozen=True)
class DailyBtXvar(
    TriageUpstreamDaily,
    HasDataFrameValue
):
    @dataclass(frozen=True)
    class Params(Daily.Params):
        unit: float = 1.
        cap: float | None = None
        key: str | None = None

        # TODO: shifts
        
        def encode(self):
            tags: list = [self.unit]
            if self.cap is not None:
                tags.append(self.cap)
            # TODO: shifts
            return "-".join(map(str, tags))

    parent: str | Object
    params: Params
    
    def get_field_df(self, k: str):
        obj: Daily | None = getattr(self, k)
        return None if obj is None else obj.get_df(k)
    


@dataclass(frozen=True)
class DailyStrategy(DailyBtXvar):

    # TODO: rx = asset
    rx: Daily | None = None
    px: Daily | None = None
    signal: Daily | None = None
    scale: Daily | None = None
    cap: Daily | None = None
    rebalance: Daily | None = None
    calendar: Daily | None = None
    fx: Daily | None = None
    gross: Daily | None = None

    # results

    bt_rx: DailyBtRx | None = None
    bt_px: DailyBtPx | None = None
    bt_pos: DailyBtPos | None = None
    bt_gross: DailyBtGross | None = None

    pfo: DailyPfolio | None = None
    
    def run_backtest(
        self,
        mv_df_scale: pl.DataFrame | None = None,
        mv_df_signal: pl.DataFrame | None = None,
        mv_df_rebalance: pl.DataFrame | None = None,
        mv_scale: float | None = None,
        mv_unit: float | None = None,
    ):
        df_rx=assert_is(self.rx).get_df("rx")

        df_scale=self.get_field_df("scale")
        df_signal=self.get_field_df("signal")
        df_rebalance=self.get_field_df("rebalance")
    
        df_scale = combine_mv_scale(
            df_scale, mv_df_scale
        )

        df_signal = combine_mv_signal(
            df_signal, mv_df_signal
        )

        df_rebalance = mv_df_rebalance or df_rebalance

        if df_scale is None and mv_scale is not None:
            df_scale = df_rx.select(
                "date", pl.lit(mv_scale).alias("scale")
            )

        df = combine_backtest_dfs(
            df_rx=df_rx,
            df_scale=df_scale,
            df_signal=df_signal,
            df_rebalance=df_rebalance,
            df_calendar=self.get_field_df("calendar"),
            df_weight=None,
            df_cap=self.get_field_df("cap"),
            df_fx=self.get_field_df("fx"),
            df_px=assert_is(self.px).get_df("px"),
            df_gross=assert_is(self.gross).get_df("gross"),
            cap=self.params.cap,
            # **shifts from params
        )

        return df.pipe(
            run_backtest,
            signal=None
            if df_signal is None
            else "signal",
            rebal=None
            if df_rebalance is None
            else "rebalance",
            calendar=None
            if self.calendar is None
            else "calendar",
            fx=None if self.fx is None else "fx",
            cap=None if self.cap is None else "cap",
            unit=self.params.unit * (
                mv_unit or 1.
            )
        )
    
    def depends_fields(self):
        return {
            k: ("ValueUpdate",)
            for k in [
                "rx",
                "px",
                "signal",
                "scale",
                "cap",
                "rebalance",
                "calendar",
                "fx",
                "gross",
            ]
        }
    
    def self(self) -> DailyStrategy:
        return self
    
    def get_parent(self) -> Object:
        return get_parent(self)

    def encode_type(self, params: DailyBtXvar.Params):
        return "bt-uv"

    def encode_params(self, params: DailyBtXvar.Params):
        return params.encode()

    def price_symbols(self):
        return rx_price_symbols(self)

    def price_types(self):
        return rx_price_types(self)

    def handle(self, event: Event, snap: Snap):
        snap = assert_is(snap, self.Snap)
        
        res = self.run_backtest()
        
        updates = frozendict()

        children: dict[str, (
            DailyBtRx
            | DailyBtPx
            | DailyBtPos
            | DailyBtGross
        )] = {
            "rx": assert_is(self.bt_rx),
            "px": assert_is(self.bt_px),
            "pos": assert_is(self.bt_pos),
            "gross": assert_is(self.bt_gross),
        }

        for k, child in children.items():
            child.set_value_from_df(res, k)
            child_snap = get_snap(child)

            updates = ValueUpdate.if_snap_changed(
                child_snap,
                child_snap.update_from_df(res),
                into=updates,
            )

        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(res),
            into=updates,
        )

# NOTE: either can keep their own id if already have one
# or, get auto assigned as basket-i

@dataclass(frozen=True)
class DailyBasket(DailyBtXvar):

    signal: Daily | None = None
    scale: Daily | None = None
    cap: Daily | None = None # unused
    rebalance: Daily | None = None
    calendar: Daily | None = None # unused
    fx: Daily | None = None # unused
    gross: Daily | None = None

    assets: tuple[DailyStrategy, ...] | None = None

    bt_rx: tuple[DailyBtRx, tuple[DailyBtRx, ...]] | None = None
    bt_px: tuple[DailyBtPx, tuple[DailyBtPx, ...]] | None = None
    bt_pos: tuple[DailyBtPos, tuple[DailyBtPos, ...]] | None = None
    bt_gross: tuple[DailyBtGross, tuple[DailyBtGross, ...]] | None = None

    pfo: DailyPfolio | None = None
    
    def depends_fields(self):
        return {
            k: ("ValueUpdate",)
            for k in [
                "assets",
                "rx",
                "px",
                "signal",
                "scale",
                "cap",
                "rebalance",
                "calendar",
                "fx",
                "gross",
            ]
        }
    
    def self(self) -> DailyBasket:
        return self
    
    def get_parent(self) -> Object:
        return get_parent(self)

    def encode_type(self, params: DailyBtXvar.Params):
        return "bt-mv"

    def encode_params(self, params: DailyBtXvar.Params):
        return params.encode()

    def price_symbols(self):
        return rx_price_symbols(self)

    def price_types(self):
        return rx_price_types(self)
    
    def handle(self, event: Event, snap: Snap):
        snap = assert_is(snap, self.Snap)

        df_scale=self.get_field_df("scale")
        df_signal=self.get_field_df("signal")
        df_rebalance=self.get_field_df("rebalance")
        df_calendar=self.get_field_df("calendar")
        # df_weight=None
        df_cap=self.get_field_df("cap")
        df_fx=self.get_field_df("fx")

        acc: list[pl.DataFrame] = []

        for uv in assert_is(self.assets):

            res = uv.run_backtest(
                mv_df_scale=df_scale,
                mv_df_signal=df_signal,
                mv_df_rebalance=df_rebalance,
                mv_scale=1.,
                mv_unit=self.params.unit,
            )

            acc.append(res)

        n = len(acc)

        res = (
            apis.pl.join_dfs(
                [
                    df.select(
                        "date", "pos", "gross", "px", "rx"
                    ).rename(
                        {
                            "gross": f"gross_{i}",
                            "px": f"px_{i}",
                            "pos": f"pos_{i}",
                            "rx": f"rx_{i}",
                        }
                    )
                    for i, df in enumerate(acc)
                ],
                align=True,
                on="date",
            )
            .with_columns(
                pl.all_horizontal(
                    [
                        pl.col(f"rx_{i}").is_null()
                        for i in range(n)
                    ]
                ).alias("mask"),
                pl.sum_horizontal(
                    *[pl.col(f"rx_{i}") for i in range(n)]
                ).alias("rx"),
                pl.all_horizontal(
                    [
                        pl.col(f"px_{i}").is_null()
                        for i in range(n)
                    ]
                ).alias("px_mask"),
                pl.sum_horizontal(
                    *[
                        pl.col(f"px_{i}").forward_fill()
                        for i in range(n)
                    ]
                ).alias("px"),
                pl.all_horizontal(
                    [
                        pl.col(f"gross_{i}").is_null()
                        for i in range(n)
                    ]
                ).alias("gross_mask"),
                pl.sum_horizontal(
                    *[
                        pl.col(f"gross_{i}").forward_fill()
                        for i in range(n)
                    ]
                ).alias("gross"),
            )
            .with_columns(
                pl.when(pl.col("mask"))
                .then(None)
                .otherwise("rx")
                .alias("rx"),
                pl.when(pl.col("px_mask"))
                .then(None)
                .otherwise("px")
                .alias("px"),
                pl.when(pl.col("gross_mask"))
                .then(None)
                .otherwise("gross")
                .alias("gross"),
            )
            .drop("mask", "px_mask", "gross_mask")
        )
        
        updates = frozendict()

        children: dict[str, (
            tuple[DailyBtRx, tuple[DailyBtRx, ...]]
            | tuple[DailyBtPx, tuple[DailyBtPx, ...]]
            | tuple[DailyBtGross, tuple[DailyBtGross, ...]]
            | tuple[DailyBtPos, tuple[DailyBtPos, ...]]
        )] = {
            "rx": assert_is(self.bt_rx),
            "px": assert_is(self.bt_px),
            "pos": assert_is(self.bt_pos),
            "gross": assert_is(self.bt_gross),
        }

        for k, (child, child_uvs) in children.items():
            assert len(child_uvs) == len(acc), self
            
            child.set_value_from_df(res, k)
            child_snap = get_snap(child)

            updates = ValueUpdate.if_snap_changed(
                child_snap,
                child_snap.update_from_df(res),
                into=updates,
            )

            for i, child_uv in enumerate(child_uvs):
                child.set_value_from_df(res, f"{k}_{i}")
                child_uv_snap = get_snap(child_uv)
                updates = ValueUpdate.if_snap_changed(
                    child_uv_snap,
                    child_uv_snap.update_from_df(res),
                    into=updates,
                )
        
        return ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_df(res),
            into=updates,
        )


#  ------------------

@dataclass(frozen=True)
class DailyPfolio(
    Daily,
    HasDataFrameValue,
):
    
    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: str | DailyStrategy | DailyBasket
    params: Params

    rx: DailyBtRx | None = None

    bt_rx: dict[str, DailyBtRx] | None = None
    bt_pos: dict[str, DailyBtPos] | None = None

    def handle(self, event: Event, snap: Snap):
        snap = assert_is(snap, self.Snap)
        
        data = assert_is(get_daily_pfolio_data(get_parent(self)))

        bt_rx = assert_is(self.bt_rx)
        bt_pos = assert_is(self.bt_pos)

        assert len(data.series) == len(bt_rx) == len(bt_pos), self

        dfs_rx = []

        updates = frozendict()

        for i, ser in enumerate(data.series):
            k = ser.key

            try:
                obj_pos = bt_pos[k]
                obj_rx = bt_rx[k]
            except Exception as e:
                raise ValueError((self, ser)) from e

            context.get().with_obj(obj_pos)
            context.get().with_obj(obj_rx)

            # log(handle_pfo=k)

            _ = obj_pos.set_value_from_df(
                ser.pos.select("date", "pos"),
                "pos",
            )

            _ = obj_rx.set_value_from_df(
                ser.rx,
                "rx",
            )

            obj_pos_snap = get_snap(obj_pos)
            obj_rx_snap = get_snap(obj_rx)
            
            updates = ValueUpdate.if_snap_changed(
                obj_pos_snap,
                obj_pos_snap.update_from_dates(data.start, data.end),
                into=updates,
            )
            updates = ValueUpdate.if_snap_changed(
                obj_rx_snap,
                obj_rx_snap.update_from_dates(data.start, data.end),
                into=updates,
            )

            dfs_rx.append(
                ser.rx.rename(
                    {"rx": f"{ser.key}={ser.rx_id}"}
                )
            )

        df_rx = apis.pl.join_dfs(
            dfs_rx,
            on="date",
            how="full",
            coalesce=True,
        ).sort(by="date")

        df_rx = df_rx.select(
            "date",
            pl.when(
                pl.all_horizontal(
                    [
                        pl.col(k).is_null()
                        for k in df_rx.columns
                        if k != "date"
                    ]
                )
            )
            .then(None)
            .otherwise(
                pl.sum_horizontal(pl.exclude("date"))
            )
            .alias("rx"),
        )

        updates = ValueUpdate.if_snap_changed(
            snap,
            snap.update_from_dates(data.start, data.end),
            into=updates,
        )
        
        pfo_rx = assert_is(self.rx)

        _ = pfo_rx.set_value_from_df(
            df_rx,
            "rx",
        )

        pfo_rx_snap = get_snap(pfo_rx)
        updates = ValueUpdate.if_snap_changed(
            pfo_rx_snap,
            pfo_rx_snap.update_from_dates(data.start, data.end),
            into=updates,
        )
        
        return updates

@dataclass(frozen=True)
class DailyBtPos(
    Daily,
    HasDataFrameValue
):
    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: str | DailyStrategy | DailyBasket
    params: Params

    def self(self) -> DailyBtPos:
        return self
    
    def get_parent(self) -> DailyStrategy | DailyBasket:
        return get_parent(self)

    def encode_type(self, params: DailyEM.Params):
        return "pos"

    def encode_params(self, params: DailyEM.Params):
        return ""

@dataclass(frozen=True)
class DailyBtRx(
    Daily,
    HasDataFrameValue
):
    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: str | DailyStrategy | DailyBasket
    params: Params

    def self(self) -> DailyBtRx:
        return self
    
    def get_parent(self) -> DailyStrategy | DailyBasket:
        return get_parent(self)

    def encode_type(self, params: DailyEM.Params):
        return "rx"

    def encode_params(self, params: DailyEM.Params):
        return ""

# TODO: perfect example of needing a bind_anon (the result rx,pos, gross, etc.)

@dataclass(frozen=True)
class DailyBtPx(
    Daily,
    HasDataFrameValue
):
    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: str | DailyStrategy | DailyBasket
    params: Params

    def self(self) -> DailyBtPx:
        return self
    
    def get_parent(self) -> DailyStrategy | DailyBasket:
        return get_parent(self)

    def encode_type(self, params: DailyEM.Params):
        return "px"

    def encode_params(self, params: DailyEM.Params):
        return ""


@dataclass(frozen=True)
class DailyBtGross(
    Daily,
    HasDataFrameValue
):
    @dataclass(frozen=True)
    class Params(Daily.Params):
        pass

    parent: str | DailyStrategy | DailyBasket
    params: Params

    def self(self) -> DailyBtGross:
        return self
    
    def get_parent(self) -> DailyStrategy | DailyBasket:
        return get_parent(self)

    def encode_type(self, params: DailyEM.Params):
        return "gross"

    def encode_params(self, params: DailyEM.Params):
        return ""



#  ------------------



__all__ = [
    "DailyStrategy",
    "DailyBasket",
    "DailyBtPos",
    "DailyBtRx",
    "DailyBtPx",
    "DailyBtGross",
]
