
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
    i: int
    bt: str
    rx: str
    perf: str
    pos: str

@dataclass(frozen=True)
class DailyBtUvarResult(BtResult):
    res: DailyBtUvarValues | None = None

    def get_bt(
        self,
        domain: Domain,
        univ: UniverseDaily,
        # TODO: resacle
    ):
        assert self.res is not None, self
        return domain.get(self.res.bt, DailyBtUvar)

    def get_pos(self, domain: Domain):
        assert self.res is not None, self
        return domain.get(self.res.pos, Daily)
    def get_rx(self, domain: Domain):
        assert self.res is not None, self
        return domain.get(self.res.rx, Daily)
    def get_perf(self, domain: Domain):
        assert self.res is not None, self
        return domain.get(self.res.perf, Daily)
    
    def into_uvar(
        self,
        univ: UniverseDaily,
        bt: DailyBtUvar | None = None,
    ):
        assert isinstance(self, DailyBtUvarResult), self
        assert isinstance(univ, UniverseDaily), univ
        if bt is None:
            assert self.i is not None, self
            univ, i = univ.with_uvar(BtUvar(self.i))
            bt = DailyBtUvar.new(univ, i)
            self = self.with_obj(bt)
        bt_pos = DailyBtPos.new(bt)
        bt_rx = DailyBtRx.new(bt)
        bt_bench = DailyBtBench.new(bt_rx)
        self = replace(
            self,
            res = DailyBtUvarValues(
                bt.index,
                bt.path,
                bt_rx.path,
                bt_bench.path,
                bt_pos.path,
            )
        ).with_obj((bt_pos, bt_rx, bt_bench))
        return univ, self

    
@dataclass(frozen=True)
class DailyBtMvarValues(BtValues):
    i: int
    bt: str
    rx: str
    perf: str
    i_rx: tuple[str, ...]
    i_pos: tuple[str, ...]
    i_perf: tuple[str, ...] | None = None
    i_signal: tuple[str, ...] | None = None
    i_scale: tuple[str, ...] | None = None

@dataclass(frozen=True)
class DailyBtMvarResult(BtResult):
    res: DailyBtMvarValues | None = None

    def get_bt(
        self,
        domain: Domain,
        univ: UniverseDaily,
    ):
        assert self.res is not None, self
        return domain.get(self.res.bt, DailyBtMvar)

    def get_pos(self, domain: Domain, i: int):
        assert self.res is not None, self
        return domain.get(self.res.i_pos[i], Daily)
    
    def get_rx(self, domain: Domain, i: int | None = None):
        assert self.res is not None, self
        if i is None:
            return domain.get(self.res.rx, Daily)
        return domain.get(self.res.i_rx[i], Daily)
    
    def get_perf(self, domain: Domain):
        assert self.res is not None, self
        return domain.get(self.res.perf, Daily)

    def into_mvar(
        self,
        univ: UniverseDaily,
        bts: list[int],
        bt: DailyBtMvar | None = None,
        with_i_perf: bool = False,
        with_i_signal: bool = False,
        with_i_scale: bool = False,
    ):
        assert isinstance(self, DailyBtMvarResult), self
        assert isinstance(univ, UniverseDaily), univ
        # TODO: dont need to bind the other objects if going to rescale?
        if bt is None:
            assert self.i is not None, self
            univ, i = univ.with_mvar(
                BtMvar(self.i, tuple(bts))
            )
            bt = DailyBtMvar.new(univ, i)
            self = self.with_obj(bt)
        bt_rx = DailyBtRx.new(bt)
        bt_bench = DailyBtBench.new(bt_rx)
        bts_pos = [
            DailyBtPos.new(bt, index=i)
            for i in range(len(bts))
        ]
        bts_rx = [
            DailyBtRx.new(bt, index=i)
            for i in range(len(bts))
        ]
        objs: tuple[Object, ...] = (
            (bt_rx, bt_bench)
            + tuple(bts_pos)
            + tuple(bts_rx)
        )
        i_perf = None
        i_signal = None
        i_scale = None
        if with_i_perf:
            bts_perf = tuple([
                DailyBtBench.new(rx) for rx in bts_rx
            ])
            objs = objs + bts_perf
            i_perf = tuple([o.path for o in bts_perf])
        if with_i_signal:
            bts_signal = tuple([
                univ.assume_one_signal(i, Daily)
                for i in bts
            ])
            objs = objs + bts_signal
            i_scale = tuple([o.path for o in bts_signal])
        if with_i_scale:
            bts_scale = tuple([
                univ.assume_one_signal(i, Daily) 
            ])
            objs = objs + bts_scale
            i_signal = tuple([o.path for o in bts_scale])
        self = replace(
            self,
            res = DailyBtMvarValues(
                bt.index,
                bt.path,
                bt_rx.path,
                bt_bench.path,
                tuple((rx.path for rx in bts_rx)),
                tuple((pos.path for pos in bts_pos)),
                i_perf = i_perf,
                i_signal = i_signal,
                i_scale = i_scale,
            )
        ).with_obj(objs)
        return univ, self

__all__ =  [
    "DailyBtUvarValues",
    "DailyBtUvarResult",
    "DailyBtMvarValues",
    "DailyBtMvarResult",
]