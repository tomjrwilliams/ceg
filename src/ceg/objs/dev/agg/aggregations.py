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
    TypeAlias,
    overload,
    cast,
    final,
)
from pathlib import Path
import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

import datetime as dt
from frozendict import frozendict

import numpy as np
import polars as pl

from ....engine import *

NAME = "Aggregation"

class Interface(BaseInterface):

    @dataclass(frozen=True)
    class Args(BaseInterface.Args):
        parent: Project | str | None

        params: tuple[str,  ...] | t.frozen.dict[str, str]

    @dataclass(frozen=True)
    class Params(Args):

        uid: str | None = None
        xid: int | str | None = None

    @dataclass(frozen=True)
    class Object(
        BaseInterface.Object,
        Args,
        DependsParamsValue,
    ):

        @dataclass(frozen=True)
        class Snap(TriageUpstream.Snap):
            type: Literal["Aggregation"] = NAME
        
        def handler(self, event: Event, snap: Snap) -> Handler:
            if isinstance(event, ValueUpdate):
                return Interface.Handler()
            # elif isinstance(event, ValueUpdate):
            raise ValueError((self, event))
        
    @dataclass(frozen=True)
    class Handler(
        # HandleNewValueFromPl,
        # HandleNewDailyFromDf,
    ):
        def handle(
            self,
            event: ValueUpdate,
            snap: Aggregation.Snap,
            obj: Aggregation,
        ):

            # return ValueUpdate.from_snap(
            #     self.update_snap_from_df(snap_new, df),
            #     into=frozendict(),
            # )
            return

@dataclass(frozen=True)
class Aggregation(Interface.Object):
    
    version: int = 0
    type: Literal["Aggregation"] = NAME

    @dataclass(frozen=True)
    class Snap(Interface.Object.Snap):
        pass

new = init_new(Interface.Params, Aggregation)

Type: TypeAlias = Aggregation

__all__ = ["Interface", "Type", "new"]