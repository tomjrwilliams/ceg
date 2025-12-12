
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

from ..root import *

#  ------------------


def apply_activation(
    ser: pl.Series,
    activation: str | None = None
):
    if activation is None:
        return ser.fill_nan(None)
    activation = (
        activation.replace("ng", "neg")
        .replace("tnlog", "tanhlog")
    )
    if activation == "tanh":
        return ser.tanh().fill_nan(None)
    elif activation == "unit_tanh":
        return (1 + ser.tanh().fill_nan(None)) / 2
    elif activation == "neg":
        return (-1 * ser).fill_nan(None)
    elif activation == "neg_tanhlog":
        return (
            -1 * ser.log().tanh()
        ).fill_nan(None)
    elif activation == "neg_tanhlog_neg":
        return (
            -1 * (-1 * ser).log().tanh()
        ).fill_nan(None)
    elif activation == "n_tanh_p_one":
        return (
            pl.DataFrame({VALUE: ser.tanh()})
            .select(
                pl.when(pl.col(VALUE).is_null())
                .then(None)
                .when(pl.col(VALUE) < 0)
                .then(pl.col(VALUE))
                .otherwise(1)
                .alias(VALUE)
            )
            .get_column(VALUE).fill_nan(None)
        )
    elif activation == "n_one_p_tanh":
        return (
            pl.DataFrame({VALUE: ser.tanh()})
            .select(
                pl.when(pl.col(VALUE).is_null())
                .then(None)
                .when(pl.col(VALUE) > 0)
                .then(pl.col(VALUE))
                .otherwise(-1)
                .alias(VALUE)
            )
            .get_column(VALUE).fill_nan(None)
        )
    else:
        raise ValueError((ser, activation))


def get_value(
    obj: Daily | list[Daily],
    domain: Domain,
    scale: float = 1.0,
    method: str | None = None,
    activation: str | None = None,
):
    if isinstance(obj, Object):
        s = (
            obj.value_key_val(domain)
            .get(obj.path, ValuePlSeries)
        )
        assert s is not None, s
        ser = s.ser
        ser = apply_activation(ser, activation) * scale
        snap = obj.snap_key_val(domain).get(
            obj.path, DailySnap
        )
        assert snap is not None, obj
        return ValuePlSeries.synthetic(
            ser, start=snap.first, end=snap.last
        )
    vs = [
        o.value_key_val(domain)
        .get(o.path, ValuePlSeries)
        for o in obj
    ]
    snap = obj[0].snap_key_val(domain).get(
        obj[0].path, DailySnap
    )
    assert snap is not None, obj

    # TODO: df join if given diff dates by snap
    sers = [
        v.ser.alias(v.obj) for v in vs if v is not None
    ]
    assert len(sers) == len(vs), (obj, len(sers), len(vs))

    df = pl.DataFrame()
    if method == "sum":
        df = df.select(
            pl.sum_horizontal(pl.all()).alias("res")
        )
    elif method == "mean":
        df = df.select(
            pl.mean_horizontal(pl.all()).alias("res")
        )
    elif method == "prod":
        df = df.select(
            pl.fold(
                acc=pl.lit(1), 
                function=lambda acc, x: acc * x, 
                exprs=pl.all(),
                return_dtype=pl.Float64,
            )
            .alias("res")
        )
    else:
        assert len(obj) == 1, obj
        assert len(vs) == 1, obj
        v = vs[0]
        assert v is not None, obj
        df = df.select(
            pl.col(v.obj).alias("res")
        )
    ser = df.get_column("res")
    ser = apply_activation(ser, activation) * scale
    return ValuePlSeries.synthetic(
        ser, start=snap.first, end=snap.last
    )

@dataclass(frozen=True)
class DailyBtValues(BtValues):
    i: int
    pos: str
    rx: str

@dataclass(frozen=True)
class DailyBtResult(BtResult):
    res: DailyBtValues | None = None
    
@dataclass(frozen=True)
class UniverseDaily(Universe):

    type: Literal["UniverseDaily"] = "UniverseDaily"

    def finish_bt(
        self,
        res: BtResult
    ):
        assert isinstance(res, DailyBtResult), res
        assert isinstance(self, UniverseDaily), self
        bt = DailyBt.new(self, res.i_bt)
        bt_pos = DailyBtPos.new(bt)
        bt_rx = DailyBtRx.new(bt)
        res = replace(
            res,
            res = DailyBtValues(
                res.i_bt,
                bt_pos.path,
                bt_rx.path,
            )
        ).with_obj((bt_pos, bt_rx))
        return res

    def scale_value(
        self, i: int | UniverseScale, domain: Domain, 
    ) -> ValuePlSeries:
        if isinstance(i, int):
            obj = self.get_scale(i, domain, Daily)
            return get_value(
                obj,
                domain,
            )
        objs = [
            self.get_scale(ii, domain, Daily)
            for ii in i.index
        ]
        return get_value(
            objs,
            domain,
            method=i.method,
            scale=i.scale,
            activation=i.activation,
        )

    def signal_value(
        self, i: int | UniverseSignal, domain: Domain, 
    ) -> ValuePlSeries:
        if isinstance(i, int):
            obj = self.get_signal(i, domain, Daily)
            assert obj is not None, obj
            return get_value(
                obj,
                domain,
            )
        objs = [
            self.get_signal(ii, domain, Daily)
            for ii in i.index
        ]
        assert all([obj is not None for obj in objs]), objs
        return get_value(
            objs, # type: ignore
            domain,
            method=i.method,
            scale=i.scale,
            activation=i.activation,
        )

    def rebalance_value(
        self, 
        i: int | str | None, 
        domain: Domain, 
        value: Daily | None = None,
    ):
        if i is None:
            return None
        elif isinstance(i, int):
            obj = self.get_rebalance(i, domain, Daily)
            assert obj is not None, i
            v = (
                obj.value_key_val(domain)
                .get(obj.path, ValuePlSeries)
            )
            return v
        else:
            assert value is not None, value
            snap = value.get_snap(domain)
            assert isinstance(snap, DailySnap), snap
            df = pl.select(
                pl.date_range(snap.first, snap.last)
                .alias("date")
            ).select(
                apis.cal.rebalance_mask(i)
                .alias("mask")
            )
            return ValuePlSeries.synthetic(
                df.get_column("mask")
            )
        
X = "x"

# TODO: separate objects for cum, daily path dependent, etc.

@dataclass(frozen=True)
class DailyBt(
    UseRootConfigs,
    DependsSelfNew,
    DependsDomainSync,
):
    parent: UniverseDaily | str

    index: int # of a bt line in parent

    version: int = 0
    type: Literal["DailyBt"] = "DailyBt"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyBtSnap"] = "DailyBtSnap"

    def get_snap(
        self, domain: Domain
    ) -> DailyBt.Snap | None:
        return self.snap_key_val(domain).get(self.path, DailyBt.Snap)
    
    def get_parent(self, domain: Domain) -> UniverseDaily:
        return domain.get(self.parent, UniverseDaily)
    
    @classmethod
    def new(
        cls,
        parent: Universe | UniverseDaily,
        index: int,
    ):
        assert isinstance(parent, UniverseDaily), (
            parent, cls
        )
        index_pad = str(index).zfill(3)
        return cls(
            parent=parent,
            path=parent.extend_path(f"bt-{index_pad}"),
            index=index,
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )
    
    def handler_object_new(
        self
    ) -> HandleNewDailyBt:
        return HandleNewDailyBt()

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(self.handler_object_new())

        # TODO: if upstream target change

        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res
    
@dataclass(frozen=True)
class HandleNewDailyBt(
    HandleNewValueFromPl
):

    def handle(
        self,
        event: ObjectNew,
        obj: DailyBt,
        domain: Domain,
    ):
        obj_rx = DailyBtRx.new(obj)
        obj_pos = DailyBtPos.new(obj)

        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None

        univ = obj.get_parent(domain)

        bt = univ.bt[obj.index]
        pos = univ.pos[bt.pos]

        rx = domain.get(bt.rx, Daily)

        v_scale = univ.scale_value(
            pos.scale, domain
        )
        v_rebalance = univ.rebalance_value(
            pos.rebalance, domain, 
            value=rx
        )
        v_signal = (
            None if pos.signal is None 
            else univ.signal_value(pos.signal, domain)
        )

        calendar = univ.get_calendar(
            pos.calendar, domain, Daily
        )
        weight = univ.get_weight(
            pos.weight, domain, Daily
        )

        rx_snap = rx.get_snap(domain)
        assert isinstance(rx_snap, DailySnap), (rx, rx_snap)

        v_rx = (
            rx.value_key_val(domain)
            .get(rx.path, ValuePlSeries)
        )
        assert v_rx is not None, obj

        v_calendar = None if calendar is None else (
            calendar.value_key_val(domain)
            .get(calendar.path, ValuePlSeries)
        )
        v_weight = None if weight is None else (
            weight.value_key_val(domain)
            .get(weight.path, ValuePlSeries)
        )
        # TODO: parametrise out the lag

        df = pl.DataFrame({
            "rx": v_rx.ser,
            "scale": (
                v_scale.ser.forward_fill()
                .shift(1, fill_value=0)
            )
        }).with_columns(
            pl.date_range(rx_snap.first, rx_snap.last)
            .alias("date")
        )
        if v_signal is not None:
            df_sig = pl.DataFrame({
                "signal": v_signal.ser
            }).with_columns(
                pl.date_range(
                    v_signal.start, # type: ignore
                    v_signal.end, # type: ignore
                ).alias("date")
            )
            df = df.join(
                df_sig,
                on="date",
                coalesce=True,
                how="left"
            ).with_columns(
                signal=(
                    pl.col("signal").forward_fill()
                    .shift(1, fill_value=0)
                )
            )
        if v_rebalance is not None:
            df = df.with_columns(
                rebalance=v_rebalance.ser
            )
        if v_calendar is not None:
            df = df.with_columns(
                calendar=v_calendar.ser
            )
        if v_weight is not None:
            df = df.with_columns(
                weight=(
                    v_weight.ser.forward_fill()
                    .shift(1, fill_value=0)
                )
            ).with_columns(
                scale=pl.col("scale") * pl.col("weight")
            )

        df = apis.perf.with_backtest(
            df,
            ("pos", "rx"),
            cum=False,
            rx="rx",
            scale="scale",
            signal=None if pos.signal is None else "signal",
            rebal=None if v_rebalance is None else "rebalance",
            calendar=None if calendar is None else "calendar",
            unit=pos.unit
        )

        assert len(df) == (
            (rx_snap.last - rx_snap.first).days + 1
        ), (df, rx_snap)

        df_pos = df.select("pos")
        df_rx = df.select("rx")

        e_pos, _ = self.set_value_from_df(
            obj_pos,
            domain,
            df_pos,
            "pos",
            t = self.key_value_type(obj, domain),
            publish=True,
        )
        e_rx, _ = self.set_value_from_df(
            obj_rx,
            domain,
            df_rx,
            "rx",
            t = self.key_value_type(obj, domain),
            publish=True,
        )

        snap_new: DailyBt.Snap = (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                rx_snap.first,
                rx_snap.last,
            ), DailyBt.Snap)
        )

        # TODO: if numpy we assume dense, use snap to infer dates

        snap_pos = (
            obj_pos.snap_key_val(domain)
            .set(obj_pos.path, type(obj_pos).Snap(
                event,
                obj_pos.path,
                dt.datetime.now(),
                snap_new.first,
                snap_new.last,
            ))
        )
        snap_rx = (
            obj_rx.snap_key_val(domain)
            .set(obj_rx.path, type(obj_rx).Snap(
                event,
                obj_rx.path,
                dt.datetime.now(),
                snap_new.first,
                snap_new.last,
            ))
        )

        return e_pos, e_rx
    
@dataclass(frozen=True)
class DailyBtPos(
    DailyChunked,
):
    parent: DailyBt | str

    version: int = 0
    type: Literal["DailyBtPos"] = "DailyBtPos"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyBtPosSnap"] = "DailyBtPosSnap"

    def get_parent(self, domain: Domain) -> DailyBt:
        return domain.get(self.parent, DailyBt)

    @classmethod
    def new(
        cls,
        parent: DailyBt,
    ):
        return cls(
            parent=parent,
            path=parent.extend_path("pos"),
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return []
    
@dataclass(frozen=True)
class DailyBtRx(
    DailyChunked,
):
    parent: DailyBt | str

    version: int = 0
    type: Literal["DailyBtRx"] = "DailyBtRx"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["DailyBtRxSnap"] = "DailyBtRxSnap"

    def get_parent(self, domain: Domain) -> DailyBt:
        return domain.get(self.parent, DailyBt)

    @classmethod
    def new(
        cls,
        parent: DailyBt,
    ):
        return cls(
            parent=parent,
            path=parent.extend_path("rx"),
        )
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return []
    

__all__ = [
    "UniverseDaily",
    "DailyBt",
    "DailyBtPos",
    "DailyBtRx",
    "DailyBtValues",
    "DailyBtResult",
    "HandleNewDailyBt",
]