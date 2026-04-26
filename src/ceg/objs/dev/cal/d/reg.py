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

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

# TODO: how to get date range without referring to parents?

#  ------------------


@dataclass(frozen=True)
class CalDailyPattern:
    n: int
    freq: str
    then: CalDailyPattern | None = None

    def path(self):
        return f"{self.freq}{self.n}"


@dataclass(frozen=True)
class CalDailyRegular(
    Daily,
):
    parent: Project | str | None

    pattern: CalDailyPattern

    version: int = 0
    type: Literal["CalDailyRegular"] = "CalDailyRegular"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["CalDailyRegularSnap"] = (
            "CalDailyRegularSnap"
        )

    @classmethod
    def new(
        cls,
        n: int,
        freq: str,
        parent: Project | None = None,
        # TODO: then
    ):
        pattern = CalDailyPattern(n=n, freq=freq)
        return cls(
            id=cls.build_id(parent, pattern.id()),
            parent=parent,
            pattern=pattern,
        )

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ObjectChange):
            assert self.id == event.obj, (self, event)
            return HandleNewCalDailyRegular()
        raise ValueError((self, event))


@dataclass(frozen=True)
class HandleUpdateCalDailyRegular(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: CalDailyRegular,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return


@dataclass(frozen=True)
class HandleNewCalDailyRegular(Handler):
    def handle(
        self,
        event: ObjectChange,
        snap: CalDailyRegular.Snap,
        obj: CalDailyRegular,
    ):
        pattern = obj.pattern

        raise ValueError(self)
        d_min, d_max = None

        v = handlers.set_value_from_df(
            obj,
            df,
            "cal",
            chunk=kv.BODY,
        )

        return ValueUpdate.from_snap(
            self.update_snap_from_df(snap, df),
            into=frozendict(),
        )


# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw


#  ------------------
@dataclass(frozen=True)
class HasCalDailyRegularParent(Object):
    parent: CalDailyRegular | str

    def get_parent(self) -> CalDailyRegular:
        return context.get().get(
            self.parent, CalDailyRegular
        )


#  ------------------

__all__ = [
    "CalDailyRegular",
    "HandleUpdateCalDailyRegular",
    "HandleNewCalDailyRegular",
    # "HasCalDailyRegularParent",
]
# TODO: if __name__ == __main__:
# run domain by name
