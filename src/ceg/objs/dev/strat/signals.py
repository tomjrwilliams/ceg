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

from .... import apis
from ....apis import t
from ....apis import kv
from ....engine import *

from ..bars.d import *
from ..calc.d import *
from ..eco.d import *

from ..agg import *
from ..pfo import *

from .assets import *

#  ------------------


@dataclass(frozen=True)
class SignalEco:
    symbol: str
    method: str
    speed: int # span or window
    activation: str | None = "tanh"
    scale: float = 1.

    def label(self):
        s = f"{self.symbol}-{self.method}-{self.speed}"
        if self.activation is not None:
            return f"{s}-{self.activation}"
        return s
    
    def with_signal(
        self,
        params: BtParams,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        v_eco = EcoDailyObservation.new(
            eng,
            symbol=self.symbol,
            source="fred",
        )
        sig: Daily
        if self.method == "z":
            sig = DailyZ.new(
                v_eco, window=self.speed
            )
        elif self.method == "ez":
            sig = DailyEZ.new(
                v_eco, span=self.speed
            )
        elif self.method == "zw":
            sig = DailyZW.new(
                v_eco, window=self.speed
            )
        elif self.method == "ezw":
            sig = DailyEZW.new(
                v_eco, span=self.speed
            )
        elif self.method == "rz":
            sig = DailyRZ.new(
                v_eco, window=self.speed
            )
        elif self.method == "erz":
            sig = DailyERZ.new(
                v_eco, span=self.speed
            )
        elif self.method == "dema":
            sig = DailyDEMA.new(
                v_eco, span=self.speed
            )
        elif self.method == "demax":
            sig = DailyDEMax.new(
                v_eco, span=self.speed
            )
        elif self.method == "zwdemin":
            demax = DailyDEMin.new(
                v_eco, span=self.speed
            )
            sig = DailyEZW.new(
                demax, span=self.speed
            )
        elif self.method == "zwdemax":
            demax = DailyDEMax.new(
                v_eco, span=self.speed
            )
            sig = DailyEZW.new(
                demax, span=self.speed
            )
        elif self.method == "raw":
            sig = v_eco
        else:
            raise ValueError((bt))
        if self.activation is not None:
            act = self.activation
            if act == "tanh":
                sig = DailyTF.new(
                    sig, f="tanh"
                )
            elif act == "unit_tanh":
                sig = DailyTF.new(
                    sig, f="tanh", a = 1., m=0.5
                )
            elif act == "n_tanh_p_one":
                sig = DailyTF.new(
                    sig, f="tanhrsign",params=(None,1)
                )
            elif act == "ng_tnlog_ng":
                sig = DailyTF.new(
                    sig,
                    b=-1,
                    f="logtanh",
                    m=-1,
                )
            else:
                raise ValueError(self)
        univ, res = universe.with_signal(
            sig,
            into=result,
            scale=self.scale,
            # activation=self.activation,
        )
        return univ, res
        
@dataclass(frozen=True)
class SignalOsc:
    fast: int
    slow: int
    vol: int
    activation: str = "tanh"

    def label(self):
        return f"osc-{self.fast}-{self.slow}-{self.vol}-{self.activation}"

    def with_signal(
        self,
        params: BtParamsWithAsset,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        asset = params.asset
        if isinstance(asset, AssetConcrete):
            px_adj = "pct" if asset.pct else "abs"
            px = (
                FutDailyOrdinal.new(
                    eng,
                    symbol=asset.symbol,
                    ident = 1,
                    field="close",
                    adjust=px_adj,
                    source="frd"
                )
                if asset.product == "FUT"
                else EqDailyEtf.new(
                    eng,
                    symbol=asset.symbol,
                    field="close",
                    adjust="price",
                    source="frd"
                )
                if asset.product == "ETF"
                else None
            )
            assert px is not None, (params, self)
            res = result.with_obj(px)
            sig = BarDailyEOsc.new(
                px,
                fast=self.fast,
                slow=self.slow,
                vol=self.vol,
                activation=self.activation
            )
            univ, res = universe.with_signal(
                sig,
                into=res,
            )
            return univ, res
        raise ValueError(self)
    
__all__ = [
    "SignalEco",
    "SignalOsc",
]