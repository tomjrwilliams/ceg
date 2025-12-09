
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

X = "x"

# TODO: separate objects for cum, daily path dependent, etc.

@dataclass(frozen=True)
class PfolioDailyScale(PfolioScale):
    pass


@dataclass(frozen=True)
class PfolioDailyPosition(
):
    scale: int | PfolioDailyScale
    signal: int | PfolioSignal | None=None
    rebalance: int | str | None=None
    calendar: int | None=None
    weight: int | None=None

    unit: float =100.

    index: int = -1

@dataclass(frozen=True)
class PfolioDailyPositionBt():
    pos: int
    rx: Object | str
    index: int = -1

@dataclass(frozen=True)
class PfolioDailyPositionAgg():
    bt: tuple[int, ...]
    # name?
    index: int = -1

    # NOTE: if bt is empty, assume take all (for total portfolio, default)

@dataclass(frozen=True)
class PfolioDaily(
    UseRootConfigs,
):
    parent: Universe | str

    ident: int # unique id (so can have different variations for a fixed universe)

    pos: tuple[PfolioDailyPosition, ...]
    bt: tuple[PfolioDailyPositionBt, ...]
    agg: tuple[PfolioDailyPositionAgg, ...]

    version: int = 0
    type: Literal["PfolioDaily"] = "PfolioDaily"

    def get_parent(self, domain: Domain) -> Universe:
        return domain.get(self.parent, t=Universe)
    
    @classmethod
    def new(
        cls,
        parent: Universe,
        ident: int,
        pos: tuple[PfolioDailyPosition, ...] = (),
        bt: tuple[PfolioDailyPositionBt, ...] = (),
        agg: tuple[PfolioDailyPositionAgg, ...] = (),
    ):
        ident_pad = str(ident).zfill(3)
        return cls(
            parent=parent,
            path=parent.extend_path(f"d-{ident_pad}"),
            # TODO: left pad?
            ident=ident,
            pos=pos,
            bt=bt,
            agg=agg,
        )
    
    def with_agg(
        self, 
        agg: tuple[int, ...] | PfolioDailyPositionAgg,
    ):
        i = len(self.agg)
        if isinstance(agg, PfolioDailyPositionAgg):
            assert agg is None, agg
            agg = replace(agg, index=i)
        else:
            agg = PfolioDailyPositionAgg(agg, i)
        return replace(self, agg=self.agg + (agg,)), i
    
    def with_bt(
        self, 
        bt: int | PfolioDailyPositionBt,
        rx: Object | str,
    ):
        i = len(self.bt)
        if isinstance(bt, PfolioDailyPositionBt):
            assert bt is None, bt
            bt = replace(bt, index=i)
        else:
            bt = PfolioDailyPositionBt(bt, rx, i)
        return replace(self, bt=self.bt + (bt,)), i
    
    def with_pos(
        self, 
        scale: int | PfolioScale | PfolioDailyPosition,
        signal: int | PfolioSignal | None=None,
        rebalance: int | str | None=None,
        calendar: int | None=None,
        weight: int | None=None,
        unit: float =100.,
    ):
        i = len(self.pos)
        if isinstance(scale, PfolioDailyPosition):
            assert scale is None, scale
            pos = replace(scale, index=i)
        else:
            if not isinstance(scale, int):
                scale = PfolioDailyScale(
                    scale.index, scale.method
                )
            pos = PfolioDailyPosition(
                scale=scale,
                signal=signal,
                rebalance=rebalance,
                calendar=calendar,
                weight=weight,
                unit=unit,
                index=i,
            )
        return replace(self, pos=self.pos + (pos,)), i
    

@dataclass(frozen=True)
class PfolioDailyBt(
    UseRootConfigs,
    DependsSelfNew,
    DependsDomainSync,
):
    parent: PfolioDaily | str

    index: int # of a bt line in parent

    version: int = 0
    type: Literal["PfolioDailyBt"] = "PfolioDailyBt"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["PfolioDailyBtSnap"] = "PfolioDailyBtSnap"

    def get_snap(
        self, domain: Domain
    ) -> PfolioDailyBt.Snap | None:
        return self.snap_key_val(domain).get(self.path, PfolioDailyBt.Snap)
    
    def get_parent(self, domain: Domain) -> PfolioDaily:
        return domain.get(self.parent, PfolioDaily)
    
    @classmethod
    def new(
        cls,
        parent: PfolioDaily,
        index: int,
    ):
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
    ) -> HandleNewPfolioDailyBt:
        return HandleNewPfolioDailyBt()

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
class HandleNewPfolioDailyBt(Handler):

    def handle(
        self,
        event: ObjectNew,
        obj: PfolioDailyBt,
        domain: Domain,
    ):
        obj_rx = PfolioDailyRx.new(obj)
        obj_pos = PfolioDailyPos.new(obj)

        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None

        pfolio = obj.get_parent(domain)
        univ = pfolio.get_parent(domain)

        bt = pfolio.bt[obj.index]
        pos = pfolio.pos[bt.pos]

        rx = domain.get(bt.rx, DailyChunked)

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
            pos.calendar, domain, DailyChunked
        )
        weight = univ.get_weight(
            pos.weight, domain, DailyChunked
        )

        rx_snap = rx.get_snap(domain)
        assert isinstance(rx_snap, DailySnap), (rx, rx_snap)

        v_rx = (
            rx.value_key_val(domain)
            .get(rx.path, kv.BODY)
        )
        v_calendar = None if calendar is None else (
            calendar.value_key_val(domain)
            .get(calendar.path, kv.BODY)
        )
        v_weight = None if weight is None else (
            weight.value_key_val(domain)
            .get(weight.path, kv.BODY)
        )

        assert isinstance(v_rx, ValuePlSeries), v_rx
        assert v_signal is None or isinstance(v_signal, ValuePlSeries), v_signal
        assert v_calendar is None or isinstance(v_calendar, ValuePlSeries), v_calendar
        assert v_weight is None or isinstance(v_weight, ValuePlSeries), v_weight

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

        df_pos = df.select("pos")
        df_rx = df.select("rx")

        e_pos, _ = set_value_chunk_from_df(
            obj_pos,
            domain,
            df_pos,
            "pos",
            chunk=kv.BODY,
            publish=True,
        )
        e_rx, _ = set_value_chunk_from_df(
            obj_rx,
            domain,
            df_rx,
            "rx",
            chunk=kv.BODY,
            publish=True,
        )

        snap_new: PfolioDailyBt.Snap = (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                rx_snap.first,
                rx_snap.last,
            ), PfolioDailyBt.Snap)
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
class PfolioDailyPos(
    UseChunkedValues,
    UseRootConfigs,
):
    parent: PfolioDailyBt | str

    version: int = 0
    type: Literal["PfolioDailyPos"] = "PfolioDailyPos"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["PfolioDailyPosSnap"] = "PfolioDailyPosSnap"

    def get_parent(self, domain: Domain) -> PfolioDailyBt:
        return domain.get(self.parent, PfolioDailyBt)

    @classmethod
    def new(
        cls,
        parent: PfolioDailyBt,
    ):
        return cls(
            parent=parent,
            path=parent.extend_path("pos"),
        )
    
@dataclass(frozen=True)
class PfolioDailyRx(
    UseChunkedValues,
    UseRootConfigs,
):
    parent: PfolioDailyBt | str

    version: int = 0
    type: Literal["PfolioDailyRx"] = "PfolioDailyRx"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["PfolioDailyRxSnap"] = "PfolioDailyRxSnap"

    def get_parent(self, domain: Domain) -> PfolioDailyBt:
        return domain.get(self.parent, PfolioDailyBt)

    @classmethod
    def new(
        cls,
        parent: PfolioDailyBt,
    ):
        return cls(
            parent=parent,
            path=parent.extend_path("rx"),
        )
    
@dataclass(frozen=True)
class DailyBtParams:
    pass

@dataclass(frozen=True)
class PfolioDailyBtRes:
    i: int
    bt: PfolioDailyBt
    pos: PfolioDailyPos
    rx: PfolioDailyRx

@dataclass(frozen=True)
class DailyBtObjects:
    params: DailyBtParams
    rx: Object
    scale: Object | tuple[Object, ...]
    signal: Object | None
    rebalance: str | None ## TODO: or object?
    # TODO: weight

    res: PfolioDailyBtRes

TDailyBtParams = TypeVar(
    "TDailyBtParams", bound=DailyBtParams
)
TDailyBtObjects = TypeVar(
    "TDailyBtObjects", bound=DailyBtObjects
)

@dataclass(frozen=True)
class DailyBtDomain(Domain):
    
    def with_bt_objs(
        self,
        universe: Universe,
        pfolio: PfolioDaily,
        params: TDailyBtParams,
        rx_bt: Object,
        scale: Object | tuple[str, tuple[Object, ...]],
        t: Callable[
            [
                TDailyBtParams,
                Object,
                Object | tuple[Object, ...], # scale
                Object | None, # signal
                str | None, # rebalance
                # rescale ?
                int,
                PfolioDailyBt,
                PfolioDailyPos,
                PfolioDailyRx,
            ],
            TDailyBtObjects
        ],
        signal: Object | tuple[Object, float, str | None] | None=None,
        rebalance: str | None = None,
        unit: float = 100.,
    ) -> tuple[
        DailyBtDomain,
        Universe,
        PfolioDaily,
        TDailyBtObjects,
    ]:
        if isinstance(scale, tuple):
            method, scales = scale
            i_scales = []
            for sc in scales:
                universe, i_sc = universe.with_scale(sc)
                i_scales.append(i_sc)
            i_scale = PfolioScale(
                tuple(i_scales),
                method=method,
            )
            scale_res = scales
        else:
            universe, i_scale = universe.with_scale(scale)
            scale_res = scale
        if isinstance(signal, tuple):
            signal, signal_scale, activation = signal
            universe, i_sig = universe.with_signal(signal)
            i_sig = PfolioSignal(
                (i_sig,),
                scale=signal_scale,
                activation=activation
            )
            pfolio, i_pos = pfolio.with_pos(
                scale=i_scale,
                signal=i_sig,
                unit=unit,
                rebalance=rebalance,
            )
        elif signal is not None:
            universe, i_sig = universe.with_signal(signal)
            pfolio, i_pos = pfolio.with_pos(
                scale=i_scale,
                signal=i_sig,
                unit=unit,
                rebalance=rebalance,
            )
        else:
            pfolio, i_pos = pfolio.with_pos(
                scale=i_scale,
                unit=unit,
                rebalance=rebalance,
            )
        pfolio, i_bt = pfolio.with_bt(
            i_pos, rx = rx_bt
        )
        bt = PfolioDailyBt.new(
            pfolio, i_bt
        )
        pos = PfolioDailyPos.new(bt)
        rx = PfolioDailyRx.new(bt)
        res = t(
            params,
            rx_bt,
            scale_res,
            signal,
            rebalance,
            i_bt,
            bt,
            pos,
            rx,
        )
        self = self.with_objs(
            res.rx,
            *(
                res.scale
                if isinstance(res.scale, tuple)
                else (res.scale,)
            ),
            res.signal,
            res.res.bt,
            res.res.pos,
            res.res.rx,
            t=DailyBtDomain,
        )
        return self, universe, pfolio, res

__all__ = [
    "PfolioDailyPosition",
    "PfolioDailyPositionBt",
    "PfolioDailyPositionAgg",
    "PfolioDaily",
    "PfolioDailyBt",
    "HandleNewPfolioDailyBt",
    "PfolioDailyPos",
    "PfolioDailyRx",
    "DailyBtParams",
    "DailyBtObjects",
    "DailyBtDomain",
    "PfolioDailyBtRes",
]