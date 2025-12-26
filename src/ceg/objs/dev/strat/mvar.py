
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
from .scales import *
from .signals import *
from .uvar import *

#  ------------------

@dataclass(frozen=True)
class BasketParams(BtParamsWithAsset):
    asset: tuple[StratParams, ...]
    name: str
    # always strat, even if basket of basket (wrap the basket in a strat, so we have scale etc.)
    scale: ScaleNotional | ScaleVol | None = None
    signal: SignalOsc | SignalEco | None = None
    rescale: ScaleVol | None = None
    rebal: str | None = None

    def label(self):
        return (
            f"{self.name}" + (
                "" 
                if self.scale is None 
                else f"-{self.scale.label()}"
            )
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
class BasketResult(DailyBtMvarResult):
    params: BasketParams

    # TODO: acc paths for sub signals?

    def vis_spec_perf(self):
        return (
            self.assume_res().perf,
            self.params.label(), 
            dict(group="perf")
        )
    
    def vis_spec_rx(self, i: int | None = None):
        if i is None:
            return (
                self.assume_res().rx,
                self.params.label(), 
                dict(group="rx")
            )
        return (
            self.assume_res().i_rx[i],
            self.params.label(), 
            dict(group="rx")
        )
    
    def vis_spec_signal(self):
        assert isinstance(self.signal, str), self
        return (
            self.signal,
            self.params.label(), 
            dict(group="signal")
        )
    
    def assume_res(self):
        assert self.res is not None, self
        return self.res

    def cast_self(
        self, univ_result: tuple[Universe, BtResult]
    ) -> tuple[UniverseDaily, BasketResult]:
        univ, res = univ_result
        assert isinstance(univ, UniverseDaily), univ
        assert isinstance(res, BasketResult), res
        return univ, res
    
    def with_bt(
        self,
        univ: UniverseDaily,
        eng: EngineRoot
    ):
        params = self.params
        cast_self = self.cast_self
    
        # univ, self = cast_self(params.asset.with_rx(
        #     params,
        #     self,
        #     univ,
        #     eng,
        # ))
        if params.scale is not None:
            univ, self = cast_self(params.scale.with_scale(
                params,
                self,
                univ,
                eng,
            ))
        if params.signal is not None:
            univ, self = cast_self(params.signal.with_signal(
                params,
                self,
                univ,
                eng,
            ))

        univ, self = cast_self(univ.with_bt(
            self, 
            rebalance=self.rebalance,
            unit=1.,
            key=None
        ))

        return univ, self

__all__ = [
    "BasketParams",
    "BasketResult",
]