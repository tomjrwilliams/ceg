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

from .etf import EqDailyEtf, DependsEqDailyEtfParent
from .stock import EqDailyStock, DependsEqDailyStockParent

#  ------------------


@dataclass(frozen=True)
class EqDailyRx(
    DependsEqDailyEtfParent,
    BarDaily,
):
    """
    >>> ord = EqDailyEtf.new(
    ...     symbol="SPY",
    ...     field="close",
    ...     adjust="full",
    ...     source="frd"
    ... )
    >>> obj = EqDailyRx.new(ord)
    >>> Graph.with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.id).tail(8)
    shape: (8,)
    Series: 'test-eq-d-SPY-close-full-frd-rx' [f64]
    [
        9.375071
        6.945601
        null
        5.529652
        null
        null
        -4.650274
        1.877995
    ]
    """

    parent: EqDailyEtf | EqDailyStock | str
    # TODO: or stock

    version: int = 0
    type: Literal["EqDailyRx"] = "EqDailyRx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["EqDailyRxSnap"] = "EqDailyRxSnap"

    @classmethod
    def new(
        cls,
        parent: EqDailyEtf | Project | None = None,
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
            parent = EqDailyEtf.new(
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
            return HandleUpdateEqDailyRx()
        elif isinstance(event, ValueUpdate):
            assert self.get_parent().id == event.obj, (
                self,
                event,
            )
            return HandleNewEqDailyRx()
        raise ValueError((self, event))


@dataclass(frozen=True)
class HandleUpdateEqDailyRx(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: EqDailyRx,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return


@dataclass(frozen=True)
class HandleNewEqDailyRx(HandleNewDailyFromDailyParentPl):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: EqDailyRx,
    ) -> pl.DataFrame:

        parent = obj.get_parent()

        # split: adjust by split ratio
        # dividends: 1 - (Dividend Amount / Prior Day's Close Price)

        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if parent.tags.adjust in ["pct", "split"]
            else pl.col(VALUE).forward_fill().diff()
            if parent.tags.adjust
            in ["raw", "price", "full"]
            # TODO: split only, raw
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
class HasEqDailyGrandParent(Object):
    parent: EqDailyEtf | EqDailyRx | str

    def get_parent(self) -> EqDailyEtf | EqDailyRx:
        return context.get().get(
            self.parent, EqDailyEtf | EqDailyRx
        )


#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "EqDailyRx",
    "HandleUpdateEqDailyRx",
    "HandleNewEqDailyRx",
    "HasEqDailyGrandParent",
]
