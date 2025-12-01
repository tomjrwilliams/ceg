
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
class MomEOscFutBtParams:
    symbol: str
    pct: bool = False
    scale: EOscScaleNotionalParams | EOscScaleVolParams = EOscScaleNotionalParams()
    signal: EOscSignalParams | None = None

    def label(self):
        pct = "-pct" if self.pct else ""
        return (
            f"{self.symbol}{pct}-{self.scale.label()}"
            + (
                "" 
                if self.signal is None 
                else f"-{self.signal.label()}"
            )
        )

@dataclass(frozen=True)
class MomEOscFutBtObjects:
    symbol: str
    params: MomEOscFutBtParams
    fut_px: FutDailyOrdinal
    fut_rx: FutDailyRx
    scale: FutDailyOrdinal | BarDailyERms
    signal: Object | None
    bt: PfolioDailyBt
    pos: PfolioDailyPos
    rx: PfolioDailyRx

@dataclass(frozen=True)
class StratMomEOscFut(Domain):
    """
    >>> spec = MomEOscFutBtParams(
    ...     "ES", 
    ...     signal=EOscSignalParams(32, 96, 48),
    ...     scale=EOscScaleVolParams(span=48)
    ... )
    >>> dom, (res,) = StratMomEOscFut.new(spec)
    >>> dom = dom.sync()
    >>> pos = res.pos
    >>> rx = res.rx
    >>> v_pos = pos.value_key_val(dom).get(pos.path, 1).ser
    >>> v_rx = rx.value_key_val(dom).get(rx.path, 1).ser
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
    pos_mu 0.1831
    pos_sd 0.213
    pos_skew 0.1727
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
        ):
            return cls(
                path = parent.extend_path(
                    "strat.mom.eosc.fut"
                ),
                parent=parent
            )

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | FutDailyOrdinal
            | FutDailyRx
            | BarDailyEOsc
            | BarDailyERms
            | PfolioDailyBt
            | PfolioDailyPos
            | PfolioDailyRx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | BarDailyEOsc.Snap
            | BarDailyERms.Snap
            | PfolioDailyBt.Snap
            | PfolioDailyPos.Snap
            | PfolioDailyRx.Snap
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
        res = []
        self = cls.empty(eng)
        universe = Universe.new(
            eng, name, ident=ident
        )
        pfolio = PfolioDaily.new(universe, ident=ident)
        for bt in bts:
            symbol = bt.symbol
            px_adj = "pct" if bt.pct else "abs"
            fut_px = FutDailyOrdinal.new(
                eng,
                symbol=symbol,
                ident = 1,
                field="close",
                adjust=px_adj,
                source="frd"
            )
            fut_rx = FutDailyRx.new(
                eng,
                symbol=symbol,
                ident = 1,
                field="close",
                adjust="abs",
                source="frd"
            )
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
                scale = BarDailyERms.new(
                    fut_rx,
                    span=bt.scale.span
                )
                unit = 100. / 16 # because not annualised
            else:
                raise ValueError(bt)
            universe = universe.with_scale(scale)
            if bt.signal is not None:
                sig = BarDailyEOsc.new(
                    fut_px,
                    fast=bt.signal.fast,
                    slow=bt.signal.slow,
                    vol=bt.signal.vol,
                    activation=bt.signal.activation
                )
                universe = universe.with_signal(sig)
                pfolio = pfolio.with_pos(
                    scale=len(universe.scale) - 1,
                    signal=len(universe.signal) - 1,
                    unit=unit,
                )
            else:
                pfolio = pfolio.with_pos(
                    scale=len(universe.scale) - 1,
                    unit=unit,
                )
                sig = None
            pfolio = pfolio.with_bt(
                len(pfolio.pos) - 1, rx = fut_rx
            )
            bt_params = bt
            bt = PfolioDailyBt.new(pfolio, len(res))
            pos = PfolioDailyPos.new(bt)
            rx = PfolioDailyRx.new(bt)
            bt_res = MomEOscFutBtObjects(
                symbol=bt_params.symbol,
                params = bt_params,
                fut_px=fut_px,
                fut_rx=fut_rx,
                scale=scale,
                signal=sig,
                bt=bt,
                pos=pos,
                rx=rx,
            )
            self = self.with_objs(
                bt_res.fut_px,
                bt_res.fut_rx,
                bt_res.scale,
                bt_res.signal,
                bt_res.bt,
                bt_res.pos,
                bt_res.rx,
            )
            res.append(bt_res)
        
        self = self.with_objs(universe, pfolio, force=True)
        return self, res
