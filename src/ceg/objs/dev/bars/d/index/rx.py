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

from ...... import apis
from ......apis import t
from ......apis import kv
from ......engine import *

from ..ident import *

from .px import IndexDailyPx, DependsIndexDailyPxParent

#  ------------------


@dataclass(frozen=True)
class IndexDailyRx(
    DependsIndexDailyPxParent,
    BarDaily,
):
    """
    >>> ord = IndexDailyPx.new(
    ...     symbol="VIX",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd"
    ... )
    >>> obj = IndexDailyRx.new(ord, adjust="abs")
    >>> Graph.with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-idx-d-VIX-close-raw-frd-rx' [f64]
    [
        -0.52
        null
        null
        -1.8
        -0.33
        1.18
        0.36
        2.96
    ]
    """

    parent: IndexDailyPx | str

    adjust: str

    version: int = 0
    type: Literal["IndexDailyRx"] = "IndexDailyRx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["IndexDailyRxSnap"] = (
            "IndexDailyRxSnap"
        )

    @classmethod
    def new(
        cls,
        parent: IndexDailyPx | Project | None = None,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str = "abs",
        source: str | None = None,
        fx: str | None = None,
    ):
        if parent is None or isinstance(parent, Project):
            assert isinstance(symbol, str), symbol
            if ident is not None:
                assert isinstance(ident, int), ident
            parent = IndexDailyPx.new(
                symbol=symbol,
                ident=ident,
                field=field,
                adjust="raw",
                source=source,
                fx=fx,
                parent=parent,
            )
        assert adjust in RX_ADJUSTMENTS, adjust
        return cls(
            id=parent.extend_id("rx"),
            parent=parent,
            adjust=adjust,
        )

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueAppend):
            return HandleUpdateIndexDailyRx()
        elif isinstance(event, ValueUpdate):
            assert self.get_parent().id == event.obj, (
                self,
                event,
            )
            return HandleNewIndexDailyRx()
        raise ValueError((self, event))


@dataclass(frozen=True)
class HandleUpdateIndexDailyRx(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: IndexDailyRx,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return


@dataclass(frozen=True)
class HandleNewIndexDailyRx(
    HandleNewDailyFromDailyParentPl
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: IndexDailyRx,
    ) -> pl.DataFrame:

        parent = obj.get_parent()

        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if obj.adjust == "pct"
            else pl.col(VALUE).forward_fill().diff()
            if obj.adjust == "abs"
            else None
        )
        assert e is not None, parent.tags.adjust

        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

        return df


# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------


@dataclass(frozen=True)
class HasIndexDailyGrandParent(Object):
    parent: IndexDailyPx | IndexDailyRx | str

    def get_parent(self) -> IndexDailyPx | IndexDailyRx:
        return context.get().get(
            self.parent, IndexDailyPx | IndexDailyRx
        )


#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "IndexDailyRx",
    "HandleUpdateIndexDailyRx",
    "HandleNewIndexDailyRx",
    "HasIndexDailyGrandParent",
]
