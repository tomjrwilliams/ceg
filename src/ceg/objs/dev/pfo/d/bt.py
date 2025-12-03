
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
    signal: int | None=None
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
        signal: int | None=None,
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
        signal = univ.get_signal(
            pos.signal, domain, DailyChunked
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
        v_signal = None if signal is None else (
            signal.value_key_val(domain)
            .get(signal.path, kv.BODY)
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

        df = pl.DataFrame({
            "rx": v_rx.ser,
            "scale": (
                v_scale.ser.forward_fill()
                .shift(1, fill_value=0)
            )
        })
        if v_signal is not None:
            df = df.with_columns(
                signal=(
                    v_signal.ser.forward_fill()
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
            signal=None if signal is None else "signal",
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
    
# TODO: need separate bt implementations per asset class

# futures assert dollar returns and implicitly dollar alloc

# equities:
# given dollar fully adjusted pct return vol
# can scale by the preceding close price to get dollar vol

# then on the next day, adjust by split value
# ie. div 4 if split into four
# but also adjust up the pos

# so you both need to pre-calc the dollar vol series, including splits
# and you need to pass the splits into the backtest, to auto-adjust (rather than effectively rebalancing itself)

# and then you can just take value(t) - value(t-1) to get pnl


# on a div day, you can either take it as a pct of prev close
# apply to the value
# net off against the price change (so essentially you add back the div)

# or you can just for that day take the fully adjusted pct change, applied to the previous value
# but still change the value using the naive price change (including the div drop out), and continue from there



# alternatively i guess if you track the position and position value separately

# so you need a starting price, and then pct returns (fully adjusted?) - ie. assume the splits net out


# ie. given a starting raw price, you recompose what the dollar price would have been, without splits
# thats probably the simplest approach

# then the question is just how to deal with dividends - in your scaling term (eg. rolling dollar vol), you probably do pct vol and multiply by the unsplit synthetic price (so does include div drop, not in vol but in the t-1 unit value)

# but in your pnl you need a different rx series
# ie. your rx series and your unsplit synthetic price aren't quite equivalent
# they differ on the dividend days, where your pnl rx series will have a (fully adjusted) pct return applied to the previous close
# so you cant just sum the pnl rx series to get back to the unsplit


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
        signal: Object | None=None,
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
        if signal is not None:
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