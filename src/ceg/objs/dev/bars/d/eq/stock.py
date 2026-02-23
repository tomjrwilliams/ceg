
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

#  ------------------

@dataclass(frozen=True)
class EqDailyStockTags(BarDailyTags):
    symbol: str
    ident: None # tbc
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

# TODO: stock data not yet downloaded

@dataclass(frozen=True)
class EqDailyStock(
    # DependsGraphSync,  # TODO
    BarDaily,
):
    # """
    # >>> obj = EqDailyStock.new(
    # ...     eng,
    # ...     symbol="AAPL",
    # ...     field="close",
    # ...     adjust="full",
    # ...     source="frd",
    # ... )
    # >>> Graph().with_obj(obj).sync()
    # >>> obj.value_key_val().get(obj.path).tail(8)
    # shape: (8,)
    # Series: 'test-fut-d-AAPL-1-close-full-frd' [f64]
    # [
    #     5718.25
    #     null
    #     null
    #     5815.5
    #     5826.5
    #     5759.5
    #     5739.25
    #     5623.0
    # ]
    # """
    parent: Project | str | None

    tags: EqDailyStockTags
    
    version: int = 0
    type: Literal["EqDailyStock"] = "EqDailyStock"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["EqDailyStockSnap"] = "EqDailyStockSnap"
    
    @classmethod
    def new(
        cls,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        parent: Project | str | None = None,
    ):
        (
            ident,
            symbol,
            stock,
            field,
            adjust,
            source,
            fx,
        ) = cls.identifier(
            symbol, 
            ident, 
            field, 
            adjust, 
            source, 
            fx=fx,
            expect=STOCK
        )
        assert adjust in EQ_ADJUSTMENTS, adjust
        return cls(
            path=cls.build_path(
                parent,
                "eq-stock-d",
                ident,
                strict=False,
            ),
            parent=parent,
            tags=EqDailyStockTags(
                symbol=symbol,
                ident=stock,
                field=field,
                adjust=adjust,
                fx=fx,
                source=source,
            ),
        )


    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, GraphSync):
            return HandleNewEqDailyStock()
        # elif isinstance(event, ValueUpdate):
        raise ValueError((self, event))

@dataclass(frozen=True)
class HandleUpdateEqDailyStock(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: EqDailyStock,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewEqDailyStock(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
    def handle(
        self,
        event: ObjectChange,
        snap: EqDailyStock.Snap,
        obj: EqDailyStock,
    ):
        tags = obj.tags

        if tags.source == "frd":
            df = apis.data.frd.read_file(
                os.environ.get("FRD_DATA", "TODO"),
                folder=apis.data.frd.Folders.STOCK,
                symbol=tags.symbol,
                suffix=tags.adjust,
                snap = "full",  # full
                freq = "1day",  # 1day
                #
                start = None,
                end = None,
            ).select("date", tags.field)
        else:
            raise ValueError(obj)

        v = self.set_value_from_df(
            obj,
            df,
            tags.field,
            t = self.key_value_type(obj),
        )

        new_snap = self.update_snap_from_df(snap, df)
        if (
            new_snap.first != snap.first
            and new_snap.last != snap.last
        ):
            return ValueUpdate.from_snap(
                self.update_snap_from_df(snap, df),
                into=frozendict()
            )
        return frozendict().set(obj.path, (snap, []))

# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------

@dataclass(frozen=True)
class DependsEqDailyStockParent(DependsParentValue):
    parent: EqDailyStock | str

    def get_parent(self) -> EqDailyStock:
        return context.get().get(self.parent, EqDailyStock)

#  ------------------

__all__ = [
    "EqDailyStock",
    "HandleUpdateEqDailyStock",
    "HandleNewEqDailyStock",
    "DependsEqDailyStockParent",
]
# TODO: if __name__ == __main__:
# run domain by name