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

# TODO: drop eng from sig and just getparent as required

@dataclass(frozen=True)
class ScaleNotional:
    
    def label(self):
        return "cn"
    
    def with_scale(
        self,
        params: BtParamsWithAsset,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        if (
            isinstance(params.asset, AssetConcrete)
            and params.asset.product == "FUT"
        ):
            scale = FutDailyOrdinal.new(
                eng,
                symbol=params.asset.symbol,
                ident = 1,
                field="close",
                adjust="raw",
                source="frd"
            )
            univ, res = universe.with_scale(
                scale,
                into=result,
                scale=1.,
            )
            return univ, res
        raise ValueError(self)

@dataclass(frozen=True)
class ScaleVol:
    span: int
    # method = rms or sdev

    ann: bool = True

    rx: str | None = None

    def label(self):
        return f"cv-{self.span}"
    
    def with_scale(
        self,
        params: BtParamsWithAsset,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        assert result.domain is not None, result
        if self.rx is None:
            assert result.rx is not None, result
            rx = result.domain.get(result.rx, Daily)
        else:
            rx = result.domain.get(self.rx, Daily)
            
        scale = DailyERms.new(
            rx,
            span=self.span,
            alpha=None,
            ann=self.ann,
        )
        univ, res = universe.with_scale(
            scale,
            into=result,
            scale=1.,
        )
        return univ, res
    
    def with_rescale(
        self,
        params: BtParams,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
        **kws,
    ):
        i_sc = result.i_scale
        assert isinstance(result.scale, str), result
        assert result.res is not None, result
        assert result.domain is not None, result
        rescale = DailyERms.new(
            result.domain.get(
                result.res.rx, Daily
            ),
            span=self.span,
            ann=True,
        )
        res = result.with_obj(rescale)
        assert res.res is not None, res
        assert result.i is not None, result
        if isinstance(res, DailyBtUvarResult):
            return res.into_uvar(
                universe,
                bt=DailyBtUvar.new(
                    universe,
                    res.res.i,
                    rescale=rescale
                )
            )
        elif isinstance(res, DailyBtMvarResult):
            return res.into_mvar(
                universe,
                list(universe.mvar[res.res.i].bts),
                bt=DailyBtMvar.new(
                    universe,
                    res.res.i,
                    rescale=rescale
                ),
                **kws
            )
        else:
            raise ValueError(result)
    
__all__ = [
    "ScaleNotional",
    "ScaleVol",
]