
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

from .px import FxDailyPx, DependsFxDailyPxParent

#  ------------------


@dataclass(frozen=True)
class FxDailyRx(
    DependsFxDailyPxParent,
    BarDaily,
):
    """
    >>> ord = FxDailyPx.new(
    ...     symbol="EURUSD",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd"
    ... )
    >>> obj = FxDailyRx.new(ord, adjust="abs")
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-fx-d-EURUSD-close-raw-frd-rx' [f64]
    [
        -0.00358
        null
        null
        -0.00112
        -0.00101
        -0.00374
        0.00478
        0.00272
    ]
    """
    parent: FxDailyPx | str

    adjust: str

    version: int = 0
    type: Literal["FxDailyRx"] = "FxDailyRx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["FxDailyRxSnap"] = "FxDailyRxSnap"
 
    @classmethod
    def new(
        cls,
        parent: FxDailyPx | Project | None = None,
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
            parent = FxDailyPx.new(
                symbol=symbol,
                ident=ident,
                field=field,
                adjust="raw",
                source=source,
                fx=fx,
                parent=parent,
            )
        assert adjust in RX_ADJUSTMENTS, adjust
        # assert ident is not None, parent
        return cls(
            path=parent.extend_path("rx"),
            parent=parent,
            adjust=adjust,
        )


    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, ValueAppend):
            return HandleUpdateFxDailyRx()
        elif isinstance(event, ValueUpdate):
            assert self.get_parent().path == event.obj, (self, event)
            return HandleNewFxDailyRx()
        raise ValueError((self, event))

@dataclass(frozen=True)
class HandleUpdateFxDailyRx(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: FxDailyRx,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return


@dataclass(frozen=True)
class HandleNewFxDailyRx(
    HandleNewDailyFromDailyParentPl
):
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: FxDailyRx,
    ) -> pl.DataFrame:
        symbol = obj.get_parent().tags.symbol

        if "USD" not in symbol:
            raise ValueError(self)
        
        # usdjpy = 153
        # jpyusd = 0.0065
        # if symbol[:3] == "USD":

        # NOTE: assume always flipped at px to symbol[:3]
        n_fgn = pl.col(VALUE)
        # else:
        #     n_fgn = (
        #         1 / pl.col(VALUE)
        #     ).fill_nan(None)

        # n_dom_tmw = (
        #     n_fgn / n_fgn.shift().forward_fill()
        # ) - 1

        e = n_fgn.forward_fill().pct_change()

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
class HasFxDailyGrandParent(Object):
    parent: FxDailyPx | FxDailyRx | str

    def get_parent(self) -> FxDailyPx | FxDailyRx:
        return context.get().get(self.parent, FxDailyPx | FxDailyRx)

#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "FxDailyRx",
    "HandleUpdateFxDailyRx",
    "HandleNewFxDailyRx",
    "HasFxDailyGrandParent",
]