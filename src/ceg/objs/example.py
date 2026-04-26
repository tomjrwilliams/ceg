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

from ..engine import *

NAME = "Example"

class Interface(BaseInterface):

    @dataclass(frozen=True)
    class Args(BaseInterface.Args):
        pass

    @dataclass(frozen=True)
    class Params(Args):
        uid: str | None = None
        xid: int | str | None = None

    @dataclass(frozen=True)
    class Object(
        BaseInterface.Object,
        Args,
        # TODO: add storage and depends
    ):

        @dataclass(frozen=True)
        class Snap(Snap):
            type: Literal["Example"] = NAME
        
        def handler(self, event: Event, snap: Snap) -> Handler:
            if isinstance(event, ValueUpdate):
                # TODO: fill in waht events to litsen for
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
            snap: Example.Snap,
            obj: Example,
        ):

            # v = self.set_value_from_df(
            #     obj,
            #     df,
            #     obj.id,
            #     t=self.key_value_type(obj),
            # )
            # new_snap = self.update_snap_from_parent(
            #     snap, obj, parent=parent,
            # )

            # # if (
            # #     new_snap.first != snap.first
            # #     and new_snap.last != snap.last
            # # ):

            # return ValueUpdate.from_snap(
            #     self.update_snap_from_df(new_snap, df),
            #     into=frozendict(),
            # )

            # return frozendict().set(obj.id, (new_snap, []))
            return

@dataclass(frozen=True)
class Example(Interface.Object):
    
    version: int = 0
    type: Literal["Example"] = NAME

new = init_new(Interface.Params, Example)

Type: TypeAlias = Example

__all__ = ["Interface", "Type", "new"]