
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

#  ------------------

@dataclass(frozen=True)
class Universe(
    UseRootConfigs,
):
    parent: EngineRoot | str

    name: str
    ident: int # config specifier (used for path, and to generate on the fly the below)

    # TODO: daily chunked not just object?

    scale: tuple[Object | str, ...]
    signal: tuple[Object | str, ...] = ()
    rebalance: tuple[Object | str, ...] = ()
    calendar: tuple[Object | str, ...] = ()
    weight: tuple[Object | str, ...] = ()
    
    # rx_label, scale_label, etc.?

    # to weight, generate a new portfolio and add weight
    
    version: int = 0
    type: Literal["Universe"] = "Universe"

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, t=EngineRoot)
    
    @classmethod
    def new(
        cls,
        parent: EngineRoot,
        name: str,
        ident: int,
        scale: tuple[Object | str, ...] = (),
        signal: tuple[Object | str, ...] = (),
        rebalance: tuple[Object | str, ...] = (),
        calendar: tuple[Object | str, ...] = (),
        weight: tuple[Object | str, ...] = (),
    ):
        return cls(
            parent=parent,
            path=f"pf.{name}.{ident}",
            name=name,
            ident=ident,
            scale=scale,
            signal=signal,
            rebalance=rebalance,
            calendar=calendar,
            weight=weight,
        )
    def with_scale(self, *scale: Object | str) -> Universe:
        scale = self.scale + scale
        return replace(
            self, scale=scale
        )
    def with_signal(self, *signal: Object | str) -> Universe:
        signal = (
            () if self.signal is None else self.signal
        ) + signal
        return replace(
            self, signal=signal
        )
    def with_rebalance(self, *rebalance: Object | str) -> Universe:
        rebalance = (
            () if self.rebalance is None else self.rebalance
        ) + rebalance
        return replace(
            self, rebalance=rebalance
        )
    def with_calendar(self, *calendar: Object | str) -> Universe:
        calendar = (
            () if self.calendar is None else self.calendar
        ) + calendar
        return replace(
            self, calendar=calendar
        )
    def with_weight(self, *weight: Object | str) -> Universe:
        weight = (
            () if self.weight is None else self.weight
        ) + weight
        return replace(
            self, weight=weight
        )
    def get_scale(
        self, i: int, domain: Domain, t: Type[O]
    ) -> O:
        return domain.get(self.scale[i], t=t)
    
    def get_signal(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.signal is None:
            return None
        return domain.get(self.signal[i], t=t)
    
    def get_rebalance(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.rebalance is None:
            return None
        return domain.get(self.rebalance[i], t=t)
    
    def get_calendar(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.calendar is None:
            return None
        return domain.get(self.calendar[i], t=t)
    
    def get_weight(
        self, i: int | None, domain: Domain, t: Type[O]
    ) -> O | None:
        if i is None or self.weight is None:
            return None
        return domain.get(self.weight[i], t=t)

#  ------------------
