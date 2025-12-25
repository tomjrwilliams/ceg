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

#  ------------------


@dataclass(frozen=True)
class AssetConcrete:
    product: str
    symbol: str
    pct: bool = False

    def get_key(self):
        return self.label()

    def label(self):
        pct = "-pct" if self.pct else ""
        return f"{self.symbol}{pct}"
    
    def with_rx(
        self,
        params: BtParams,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        rx = (
            FutDailyRx.new(
                eng,
                symbol=self.symbol,
                ident = 1,
                field="close",
                adjust="abs",
                source="frd"
            )
            if self.product == "FUT"
            else EqDailyRx.new(
                eng,
                symbol=self.symbol,
                field="close",
                adjust="full",
                source="frd"
            )
            if self.product == "ETF"
            else None
        )
        assert rx is not None, self
        return universe.with_rx(
            rx, into=result
        )

@dataclass(frozen=True)
class AssetSynthetic:
    name: str
    alias: str | None = None
    
    def get_key(self):
        return None

    def label(self):
        return self.alias or self.name
    
    def with_rx(self, 
        params: BtParams,
        result: BtResult,
        universe: UniverseDaily,
        eng: EngineRoot,
    ):
        assert result.domain is not None, result
        rx = result.domain.get(self.name, t=Daily)
        assert rx is not None, self
        return universe.with_rx(
            rx, into=result
        )

    
@dataclass(frozen=True)
class BtParamsWithAsset(BtParams):
    asset: AssetConcrete | AssetSynthetic

__all__ = [
    "AssetConcrete",
    "AssetSynthetic",
    "BtParamsWithAsset"
]