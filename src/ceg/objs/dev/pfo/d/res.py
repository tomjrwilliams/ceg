
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

from ...calc import *

from ..bt import *
from .core import *

from .bt import *
from .bench import *

#  ------------------


@dataclass(frozen=True)
class DailyBtUvarValues(BtValues):
    # i: int
    pos: str
    rx: str
    perf: str

@dataclass(frozen=True)
class DailyBtUvarResult(BtResult):
    res: DailyBtUvarValues | None = None

    def into_uvar(
        self,
        univ: UniverseDaily
    ):
        assert isinstance(self, DailyBtUvarResult), self
        assert isinstance(univ, UniverseDaily), univ
        assert self.i is not None, self
        bt = DailyBtUvar.new(univ, self.i)
        bt_pos = DailyBtPos.new(bt)
        bt_rx = DailyBtRx.new(bt)
        bt_bench = DailyBtBench.new(bt_rx)
        self = replace(
            self,
            res = DailyBtUvarValues(
                # res.i,
                bt_pos.path,
                bt_rx.path,
                bt_bench.path,
            )
        ).with_obj((bt_pos, bt_rx, bt_bench))
        return univ.with_uvar(BtUvar(bt.index)), self

    
@dataclass(frozen=True)
class DailyBtMvarValues(BtValues):
    agg: str
    pos: tuple[str, ...]
    rx: tuple[str, ...]

@dataclass(frozen=True)
class DailyBtMvarResult(BtResult):
    res: DailyBtMvarValues | None = None

__all__ =  [
    "DailyBtUvarValues",
    "DailyBtUvarResult",
    "DailyBtMvarValues",
    "DailyBtMvarResult",
]