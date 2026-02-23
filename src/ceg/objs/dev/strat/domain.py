
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
import itertools
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

from ....engine import (
    Graph,
    Object, # needed bc weird pydantic subclassing
    Dependency, # needed bc weird pydantic subclassing
    Project,
    Config,
    ObjectParser,
    SnapParser,
    StorageConfig,
)

from .... import apis
from ....apis import t, kv, assert_is

from ..bars.d import *
from ..calc.d import *
from ..eco.d import *

from ..agg import *
from ..pfo import *

from .assets import *
from .calcs import *
from .params import *

K = TypeVar("K")

#  ------------------


@dataclass(frozen=True)
class StrategyGraph(Graph):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            Project
            | StorageConfig
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyRx
            | FxDailyRx
            | FxDailyPx
            | DailyEOsc
            | DailyERms
            | DailyBtPos
            | DailyBtRx
            | DailyAggregate
            | DailyEMA
            | DailyZ | DailyEZ
            | DailyRZ | DailyERZ
            | DailyZW | DailyEZW
            | DailyRZW | DailyERZW
            | DailyDEMax
            | DailyDEMin
            | DailyDEMA
            | EcoDailyObservation
            | DailyBtPerf
            | DailyTF
            | DailyPfolioPos
            | DailyPfolioRx
            | DailyBtMvar
            | DailyBtUvar
            | DailyPfolio
            | DailyPfolioNotional
            | DailyPfolioNotionalVal
            | DailyPfolioNotionalCap
            | DailyDDRegime
            | DailyEWmThold
            | Collection
            | DailyAggregateDeriv
            | Strategies
            | DailyBtWeights
            | DailyBtPx
            | DailyBtGross
            | DailyBtWeight
            # | Combination
            # | Ref
            # | Aggregate
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | EqDailyEtf.Snap
            | EqDailyRx.Snap
            | FxDailyRx.Snap
            | FxDailyPx.Snap
            | DailyEOsc.Snap
            | DailyERms.Snap
            | DailyBtPos.Snap
            | DailyBtRx.Snap
            | DailyAggregate.Snap
            | DailyAggregateDeriv.Snap
            | DailyEMA.Snap
            | DailyZ.Snap | DailyEZ.Snap
            | DailyRZ.Snap | DailyERZ.Snap
            | DailyZW.Snap | DailyEZW.Snap
            | DailyRZW.Snap | DailyERZW.Snap
            | DailyDEMin.Snap
            | DailyDEMax.Snap
            | DailyDEMA.Snap
            | EcoDailyObservation.Snap
            | Collection.Snap
            | DailyAggregate.Snap
            | Collection.Snap
            | DailyBtPerf.Snap
            | DailyTF.Snap
            | DailyPfolioPos.Snap
            | DailyPfolioRx.Snap
            | DailyBtMvar.Snap
            | DailyBtUvar.Snap
            | DailyPfolio.Snap
            | DailyPfolioNotional.Snap
            | DailyPfolioNotionalVal.Snap
            | DailyPfolioNotionalCap.Snap
            | DailyDDRegime.Snap
            | DailyEWmThold.Snap
            | DailyBtWeights.Snap
            | DailyBtPx.Snap
            | DailyBtGross.Snap
            | DailyBtWeight.Snap
        )


__all__ = [
    "StrategyGraph"
]