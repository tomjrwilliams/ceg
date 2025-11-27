
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

from .....apis import t
from .....apis import kv
from .....engine import *

from ...calc import *
from ...pfo import *

from ... import fut
from ...fut import *

#  ------------------


@dataclass(frozen=True)
class PfolioDailyDomainExample(Domain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> dom, bt = PfolioDailyDomainExample.new(eng, "ES")
    >>> pos = PfolioDailyPos.new(bt)
    >>> rx = PfolioDailyRx.new(bt)
    >>> dom = dom.with_objs(pos, rx)
    >>> dom = dom.sync()
    >>> v_pos = pos.value_key_val(dom).get(pos.path, 1).ser
    >>> v_rx = rx.value_key_val(dom).get(rx.path, 1).ser
    >>> v_pos.tail(8)
    shape: (8,)
    Series: 'pf.mom.example.ES.0-d-000-bt-000-pos' [f64]
    [
        -0.011756
        null
        null
        -0.011871
        -0.014905
        -0.015839
        -0.016008
        -0.016034
    ]
    >>> v_rx.tail(8)
    shape: (8,)
    Series: 'pf.mom.example.ES.0-d-000-bt-000-rx' [f64]
    [
        -0.064065
        null
        null
        -1.143314
        -0.13058
        0.998648
        0.320748
        1.860983
    ]
    >>> pl.DataFrame(dict(pos=v_pos, rx=v_rx)).pipe(
    ...     apis.perf.perf_row, round=4, repr=True
    ... )
    mu 3.0897
    sd 26.6913
    rms 26.6889
    skew -0.2573
    shrp 0.1167
    sort 0.1328
    dd 58.6412
    calm 0.0527
    pos_mu 0.0325
    pos_sd 0.0774
    pos_skew -0.2204
    """
    
    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

    @classmethod
    def new(
        cls,
        eng: EngineRoot,
        symbol: str,
        name: str = "mom.example",
        ident: int = 0,
        fast: int = 32,
        slow: int = 96,
        vol: int = 48,
    ):
        px_raw = FutDailyOrdinal.new(
            eng,
            symbol=symbol,
            ident = 1,
            field="close",
            adjust="raw",
            source="frd"
        )
        rx_abs = FutDailyRx.new(
            eng,
            symbol=symbol,
            ident = 1,
            field="close",
            adjust="abs",
            source="frd"
        )
        px_abs = FutDailyOrdinal.new(
            eng,
            symbol=symbol,
            ident = 1,
            field="close",
            adjust="abs",
            source="frd"
        )
        px_pct = FutDailyOrdinal.new(
            eng,
            symbol=symbol,
            ident = 1,
            field="close",
            adjust="pct",
            source="frd"
        )
        osc_pct = fut.d.m.osc.FutDailyOsc.new(
            px_pct, fast=fast, slow=slow, vol=vol
        )
        osc_abs = fut.d.m.osc.FutDailyOsc.new(
            px_abs, fast=fast, slow=slow, vol=vol
        )
        universe = (
            Universe.new(
                eng, f"{name}.{symbol}", ident=ident
            )
            .with_scale(px_raw)
            .with_signal(osc_pct)
        )
        pfolio = (
            PfolioDaily.new(universe, ident=ident)
            .with_pos(scale=0, signal=0)
            .with_bt(pos=0, rx=rx_abs)
            .with_agg(())
        )
        bt = PfolioDailyBt.new(pfolio, 0)
        # 0 ie. first backtest
        self = (
            cls.empty(eng)
            .with_objs(
                px_raw,
                rx_abs,
                px_abs,
                px_pct,
                osc_pct,
                osc_abs,
                universe,
                pfolio,
                bt,
            )
        )
        
        return self, bt

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
                    "strat.mom.example"
                ),
                parent=parent
            )

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | fut.d.m.osc.FutDailyOsc
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | fut.d.m.osc.FutDailyOsc.Snap
        )