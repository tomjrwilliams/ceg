
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

#  ------------------

@dataclass(frozen=True)
class StratParams(BtParamsWithAsset):

    # TODO: even basket result?
    asset: AssetConcrete | AssetSynthetic | StratResult
    scale: ScaleNotional | ScaleVol | None = None
    signal: SignalOsc | SignalEco | None = None
    rescale: ScaleVol | None = None
    rebal: str | None = None

    def label(self):
        assert self.scale is not None, self
        return (
            f"{self.asset.label()}-{self.scale.label()}"
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
class StratResult(DailyBtUvarResult):
    params: StratParams

    def with_rx(self, 
        params: BtParams,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        assert self.res is not None, self
        assert result.domain is not None, result
        rx = result.domain.get(self.res.rx, t=Daily)
        assert rx is not None, self
        return universe.with_rx(
            rx, into=result
        )
    
    def get_key(self):
        return None

    def label(self):
        return self.params.label()

    def assume_res(self):
        assert self.res is not None, self
        return self.res

    def cast_self(
        self, univ_result: tuple[Universe, BtResult]
    ) -> tuple[UniverseDaily, StratResult]:
        univ, res = univ_result
        assert isinstance(univ, UniverseDaily), univ
        assert isinstance(res, StratResult), res
        return univ, res

    def with_bt(
        self,
        univ: UniverseDaily,
        eng: EngineRoot,
        assert_scale: bool = True,
    ):
        params = self.params
        cast_self = self.cast_self

        assert isinstance(params.asset, (
            AssetConcrete
            | AssetSynthetic
            | StratResult
            # TODO: or basket?
        )), params
    
        univ, self = cast_self(params.asset.with_rx(
            params,
            self,
            univ,
            eng,
        ))
        if assert_scale:
            assert params.scale is not None, self
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
            key=self.params.asset.get_key()
        ))

        return univ, self

__all__ = [
    "StratParams",
    "StratResult",
]