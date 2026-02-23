
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
from .....apis import assert_is
from .....apis import t
from .....apis import kv
from .....engine import *

from ...calc import *


from .core import *
from .bt import *
from .pfo import *

#  ------------------

@dataclass(frozen=True)
class DailyBtUvarDeriv(
    Daily,
    UseRootStorage,
    DependsParentValue
):
    parent: DailyBtUvar | str

    version: int = 0
    type: Literal["TBC"] = "TBC"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["TBC"] = "TBC"

    def get_parent(self) -> DailyBtUvar:
        return context.get().get(
            self.parent, DailyBtUvar
        )
    
    
    
    def handler_object_new(
        self
    ):
        raise ValueError(self)
    
    # NOTE: implement .new()
    
    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            # assert self.get_parent().path == event.obj, (self, event)
            return self.handler_object_new()
        raise ValueError((self, event))
      
@dataclass(frozen=True)
class HandleNewDailyBtUvarDeriv(
    HandleNewValueFromPl
):
    def handle(
        self,
        event: ObjectChange,
        obj: DailyBtUvarDeriv,
    ):
        bt = obj.get_parent()

        raise ValueError(obj) # implement
    
#  ------------------

@dataclass(frozen=True)
class DailyBtMvarDeriv(
    Daily,
    UseRootStorage,
    DependsParentValue
):
    parent: DailyBtMvar | str

    version: int = 0
    type: Literal["TBC"] = "TBC"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["TBC"] = "TBC"

    def get_parent(self) -> DailyBtMvar:
        return context.get().get(
            self.parent, DailyBtMvar
        )
    
    def handler_object_new(
        self
    ):
        raise ValueError(self)
    
    # NOTE: implement .new()
    
    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            # assert self.get_parent().path == event.obj, (self, event)
            return self.handler_object_new()
        raise ValueError((self, event))
      
@dataclass(frozen=True)
class HandleNewDailyBtMvarDeriv(
    HandleNewValueFromPl
):
    def handle(
        self,
        event: ObjectChange,
        obj: DailyBtMvarDeriv,
    ):
        bt = obj.get_parent()

        raise ValueError(obj) # implement
    
#  ------------------


@dataclass(frozen=True)
class DailyPfolioDeriv(
    Daily,
    UseRootStorage,
    DependsParent
):
    parent: DailyPfolio | str

    version: int = 0
    type: Literal["TBC"] = "TBC"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["TBC"] = "TBC"

    def get_parent(self) -> DailyPfolio:
        return context.get().get(
            self.parent, DailyPfolio
        )
    
    def depends(self) -> list[Dependency]:
        parent = self.get_parent()
        rx = DailyPfolioRx.new(parent)
        return [
            Dependency(
                self.path,
                rx.path,
                events=("ValueUpdate",)
                # TODO: append, prepend?
            )
        ]
    
    def handler_object_new(
        self
    ):
        raise ValueError(self)
    
    # NOTE: implement .new()
    
    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueUpdate):
            # assert self.get_parent().path == event.obj, (self, event)
            return self.handler_object_new()
        raise ValueError((self, event))
      
@dataclass(frozen=True)
class HandleNewDailyPfolioDeriv(
    HandleNewValueFromPl
):
    def handle(
        self,
        event: ObjectChange,
        obj: DailyPfolioDeriv,
    ):
        pf = obj.get_parent()
        data = assert_is(get_daily_pfolio_data(pf))

        raise ValueError(obj) # implement
    
#  ------------------

__all__ = [
    "DailyBtUvarDeriv",
    "HandleNewDailyBtUvarDeriv",
    "DailyBtMvarDeriv",
    "HandleNewDailyBtMvarDeriv",
    "DailyPfolioDeriv",
    "HandleNewDailyPfolioDeriv",
]