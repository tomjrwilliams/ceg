
from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
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

from ...bars.d import *
from ...calc.d import *

from ...pfo import *

#  ------------------

@dataclass(frozen=True)
class EOscScaleNotionalParams:
    
    def label(self):
        return "cn"

@dataclass(frozen=True)
class EOscScaleVolParams:
    span: int
    # method = rms or sdev

    def label(self):
        return f"cv-{self.span}"

@dataclass(frozen=True)
class EOscSignalParams:
    fast: int
    slow: int
    vol: int
    activation: str = "tanh"

    def label(self):
        return f"osc-{self.fast}-{self.slow}-{self.vol}-{self.activation}"

@dataclass(frozen=True)
class MomEOscFutBtParams(DailyBtParams):
    product: str
    symbol: str
    pct: bool = False
    scale: EOscScaleNotionalParams | EOscScaleVolParams = EOscScaleNotionalParams()
    signal: EOscSignalParams | None = None
    rescale: EOscScaleVolParams | None = None
    rebal: str | None = None

    def label(self):
        pct = "-pct" if self.pct else ""
        return (
            f"{self.symbol}{pct}-{self.scale.label()}"
            + (
                "" 
                if self.signal is None 
                else f"-{self.signal.label()}"
            )
            + (
                "" 
                if self.rescale is None 
                else f"-{self.rescale.label()}"
            )
            + (
                "" 
                if self.rebal is None 
                else f"-{self.rebal}"
            )
        )

@dataclass(frozen=True)
class MomEOscFutBtObjects(DailyBtObjects):
    params: MomEOscFutBtParams
    rx: Object
    scale: (
        EqDailyEtf | FutDailyOrdinal | DailyERms 
    ) | tuple[
        EqDailyEtf | FutDailyOrdinal | DailyERms, ...
    ]
    signal: BarDailyEOsc | None
    rebalance: str | None
    res: PfolioDailyBtRes

    @classmethod
    def new(
        cls,
        params: MomEOscFutBtParams,
        rx_bt: Object,
        scale: Object | tuple[Object, ...],
        signal: Object | None,
        rebalance: str | None,
        i_bt: int,
        bt: PfolioDailyBt,
        pos: PfolioDailyPos,
        rx: PfolioDailyRx,
    ):
        assert signal is None or isinstance(signal, BarDailyEOsc), signal
        if isinstance(scale, tuple):
            assert all([
                isinstance(sc, (
                    EqDailyEtf,
                    FutDailyOrdinal,
                    DailyERms,
                )) for sc in scale
            ]), scale
            scale = cast(
                tuple[
                    EqDailyEtf
                    | FutDailyOrdinal
                    | DailyERms, ...
                ], scale
            )
        else:
            assert isinstance(scale, (
                EqDailyEtf,
                FutDailyOrdinal,
                DailyERms,
            )), scale
        return cls(
            params=params,
            rx=rx_bt,
            scale=scale,
            signal=signal,
            rebalance=rebalance,
            res = PfolioDailyBtRes(
                i_bt,
                bt=bt,
                pos=pos,
                rx=rx,
            )
        )

    # symbol: str
    # fut_px: FutDailyOrdinal
    # fut_rx: FutDailyRx

@dataclass(frozen=True)
class StratMomEOscFut(DailyBtDomain):
    """
    >>> spec = MomEOscFutBtParams(
    ...     "FUT",
    ...     "ES", 
    ...     signal=EOscSignalParams(32, 96, 48),
    ...     scale=EOscScaleVolParams(span=48)
    ... )
    >>> dom, (res,), agg = StratMomEOscFut.new(spec)
    >>> dom = dom.sync()
    >>> pos = res.res.pos
    >>> rx = res.res.rx
    >>> v_pos = pos.value_key_val(dom).get(pos.path, 1).ser
    >>> v_rx = rx.value_key_val(dom).get(rx.path, 1).ser
    >>> v_agg = agg.value_key_val(dom).get(agg.path, 1).ser
    >>> v_pos.tail(8)
    shape: (8,)
    Series: 'pf.mom.eosc.fut.0-d-000-bt-000-pos' [f64]
    [
        -0.039443
        null
        null
        -0.042208
        -0.041717
        -0.043145
        -0.044309
        -0.04657
    ]
    >>> v_agg.tail(8)
    shape: (8,)
    Series: 'pf.mom.eosc.fut.0-d-000-rx-000' [f64]
    [
        -0.201154
        0.0
        0.0
        -3.835789
        -0.464287
        2.795036
        0.873693
        5.150923
    ]
    >>> v_rx.tail(8)
    shape: (8,)
    Series: 'pf.mom.eosc.fut.0-d-000-bt-000-rx' [f64]
    [
        -0.201154
        null
        null
        -3.835789
        -0.464287
        2.795036
        0.873693
        5.150923
    ]
    >>> pl.DataFrame(dict(pos=v_pos, rx=v_rx)).pipe(
    ...     apis.perf.perf_row, round=4, repr=True
    ... )
    mu 26.3673
    sd 70.0411
    rms 70.0531
    skew -1.3384
    shrp 0.3794
    sort 0.4375
    dd 166.3657
    calm 0.1585
    p_mu 0.1831
    p_sd 0.213
    p_skew 0.1727
    """
    
    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

    @dataclass(frozen=True)
    class Root(UseRootConfigs):

        version: int = 0
        type: Literal["PfolioDailyDomainExample"] = "PfolioDailyDomainExample"

        @classmethod
        def new(
            cls,
            parent: EngineRoot,
            name: str = "mom.eosc.fut",
        ):
            return cls(
                path = parent.extend_path(name),
                parent=parent
            )

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyRx
            | BarDailyEOsc
            | DailyERms
            | PfolioDailyBt
            | PfolioDailyPos
            | PfolioDailyRx
            | PfolioDailyMetric
            | PfolioDailyAggregate
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | EqDailyEtf.Snap
            | EqDailyRx.Snap
            | BarDailyEOsc.Snap
            | DailyERms.Snap
            | PfolioDailyBt.Snap
            | PfolioDailyPos.Snap
            | PfolioDailyRx.Snap
            | PfolioDailyAggregate.Snap
        )

    @classmethod
    def new(
        cls,
        *bts: MomEOscFutBtParams,
        name: str = "mom.eosc.fut",
        ident: int = 0,
        eng: EngineRoot | None = None,
    ):
        if eng is None:
            eng = EngineRoot.new(ROOT)

        res: list[MomEOscFutBtObjects] = []
        
        self = cls.empty(eng)
        universe = Universe.new(
            eng, name, ident=ident
        )

        pfolio = PfolioDaily.new(universe, ident=ident)

        for bt in bts:
            symbol = bt.symbol
            px_adj = "pct" if bt.pct else "abs"
            px = (
                FutDailyOrdinal.new(
                    eng,
                    symbol=symbol,
                    ident = 1,
                    field="close",
                    adjust=px_adj,
                    source="frd"
                )
                if bt.product == "FUT"
                else EqDailyEtf.new(
                    eng,
                    symbol=symbol,
                    field="close",
                    adjust="price",
                    source="frd"
                )
                if bt.product == "ETF"
                else None
            )
            assert px is not None, bt
            self = self.with_objs(
                px, t=StratMomEOscFut
            )
            rx = (
                FutDailyRx.new(
                    eng,
                    symbol=symbol,
                    ident = 1,
                    field="close",
                    adjust="abs",
                    source="frd"
                )
                if bt.product == "FUT"
                else EqDailyRx.new(
                    eng,
                    symbol=symbol,
                    field="close",
                    adjust="full",
                    source="frd"
                )
                if bt.product == "ETF"
                else None
            )
            assert rx is not None, bt
            if isinstance(
                bt.scale, EOscScaleNotionalParams
            ):
                scale = FutDailyOrdinal.new(
                    eng,
                    symbol=symbol,
                    ident = 1,
                    field="close",
                    adjust="raw",
                    source="frd"
                )
                unit = 100.
            elif isinstance(bt.scale, EOscScaleVolParams):
                scale = DailyERms.new(
                    rx,
                    span=bt.scale.span,
                    alpha=None,
                )
                unit = 100. / 16 # because not annualised
            else:
                raise ValueError(bt)
            
            if bt.signal is not None:
                sig = BarDailyEOsc.new(
                    px,
                    fast=bt.signal.fast,
                    slow=bt.signal.slow,
                    vol=bt.signal.vol,
                    activation=bt.signal.activation
                )
            else:
                sig = None

            (
                self, universe, pfolio, bt_res
            ) = self.with_bt_objs(
                universe,
                pfolio,
                bt,
                rx,
                scale,
                t=MomEOscFutBtObjects.new,
                signal=sig,
                rebalance=bt.rebal,
                unit=unit,
            )

            if bt.rescale is None:
                res.append(bt_res)
                continue

            rescale = DailyERms.new(
                bt_res.res.rx,
                span=bt.rescale.span,
            )
            (
                self, universe, pfolio, bt_res
            ) = self.with_bt_objs(
                universe,
                pfolio,
                bt,
                rx,
                (
                    "prod", (scale, rescale)
                ),
                t=MomEOscFutBtObjects.new,
                signal=sig,
                rebalance=bt.rebal,
                unit=unit / 16,
            )
            res.append(bt_res)
        
        # TODO: return on notional is relevant when vol scaling

        pfolio, i_agg = pfolio.with_agg(
            tuple([r.res.i for r in res])
        )

        rx_metric = PfolioDailyMetric.new(pfolio, "rx")
        rx_agg = PfolioDailyAggregate.new(rx_metric, i_agg)

        self = self.with_objs(
            universe,
            pfolio,
            rx_metric,
            rx_agg,
            force=True,
            t=StratMomEOscFut
        )

        return self, res, rx_agg
