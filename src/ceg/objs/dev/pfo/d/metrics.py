
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

from ..root import *
from .bt import *

#  ------------------

@dataclass(frozen=True)
class PfolioPositionMetric():
    bt: int
    metric: Object | str
    index: int = -1

@dataclass(frozen=True)
class PfolioDailyMetric(
    UseRootConfigs,
):
    parent: PfolioDaily | str
    
    name: str
    
    metric: tuple[PfolioPositionMetric, ...]
    # calc however you see fit, eg. dv01
    # NOTE: optional: eg. if name=dd, will use returns instead

    version: int = 0
    type: Literal["PfolioDailyBtMetric"] = "PfolioDailyBtMetric"

    def get_parent(self, domain: Domain) -> PfolioDaily:
        return domain.get(self.parent, t=PfolioDaily)
    
    @classmethod
    def new(
        cls,
        parent: PfolioDaily,
        name: str,
        metric: tuple[PfolioPositionMetric, ...] = (),
    ):
        return cls(
            parent=parent,
            path=parent.extend_path(name),
            name=name,
            metric=metric,
        )

@dataclass(frozen=True)
class PfolioDailyAggregate(
    UseRootConfigs,
):
    parent: PfolioDailyMetric | str
    
    index: int # index of agg in pfolio

    version: int = 0
    type: Literal["PfolioDailyAggregate"] = "PfolioDailyAggregate"

    def get_parent(self, domain: Domain) -> PfolioDailyMetric:
        return domain.get(self.parent, t=PfolioDailyMetric)
    
    @classmethod
    def new(
        cls,
        parent: PfolioDailyMetric,
        index: int,
    ):
        index_pad = str(index).zfill(3)
        return cls(
            parent=parent,
            path=parent.extend_path(index_pad),
            index=index,
        )
    
# TODO: handler to calc

# handler is name dependent

# eg. for rx, we dont need metrics, and handler just sums
# likewise for rx_vol
# likewise for drawdown, also dont need metrics, handler sums and calcs dd (etc.)

# TODO: domain