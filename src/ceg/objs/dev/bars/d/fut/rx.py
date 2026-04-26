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
from .ord import (
    FutDailyOrdinal,
    DependsFutDailyOrdinalParent,
    BarDaily,
)

#  ------------------


@dataclass(frozen=True)
class FutDailyRx(
    DependsFutDailyOrdinalParent,
    BarDaily,
):
    """
    >>> ord = FutDailyOrdinal.new(
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = FutDailyRx.new(ord)
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx' [f64]
    [
        0.000963
        null
        null
        0.017007
        0.001891
        -0.011499
        -0.003516
        -0.020255
    ]
    """

    parent: FutDailyOrdinal | str
    # TODO: or contract

    version: int = 0
    type: Literal["FutDailyRx"] = "FutDailyRx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["FutDailyRxSnap"] = "FutDailyRxSnap"

    @classmethod
    def new(
        cls,
        parent: FutDailyOrdinal | Project | None = None,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
    ):
        if parent is None or isinstance(parent, Project):
            assert isinstance(symbol, str), symbol
            if ident is not None:
                assert isinstance(ident, int), ident
            parent = FutDailyOrdinal.new(
                symbol=symbol,
                ident=ident,
                field=field,
                adjust=adjust,
                source=source,
                fx=fx,
                parent=parent,
            )
        # assert ident is not None, parent
        return cls(
            id=parent.extend_id("rx"),
            parent=parent,
        )

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueAppend):
            return HandleUpdateFutDailyRx()
        elif isinstance(event, ValueUpdate):
            assert self.get_parent().id == event.obj, (
                self,
                event,
            )
            return HandleNewFutDailyRx()
        raise ValueError((self, event))


@dataclass(frozen=True)
class HandleUpdateFutDailyRx(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: FutDailyRx,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return


@dataclass(frozen=True)
class HandleNewFutDailyRx(HandleNewDailyFromDailyParentPl):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: FutDailyRx,
    ) -> pl.DataFrame:

        parent = obj.get_parent()

        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if parent.tags.adjust == "pct"
            else pl.col(VALUE).forward_fill().diff()
            if parent.tags.adjust == "abs"
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
class HasFutDailyGrandParent(Object):
    parent: FutDailyOrdinal | FutDailyRx | str

    def get_parent(self) -> FutDailyOrdinal | FutDailyRx:
        return context.get().get(
            self.parent, FutDailyOrdinal | FutDailyRx
        )


#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "FutDailyRx",
    "HandleUpdateFutDailyRx",
    "HandleNewFutDailyRx",
    "HasFutDailyGrandParent",
]
