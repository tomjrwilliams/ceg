
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
class FxDailyTags(BarDailyTags):
    symbol: str
    ident: None
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to


@dataclass(frozen=True)
class FxDailyPx(
    DependsGraphSync,
    BarDaily,
):
    """
    >>> obj = FxDailyPx.new(
    ...     symbol="EURUSD",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd",
    ... )
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-fx-d-EURUSD-close-raw-frd' [f64]
    [
        1.08118
        null
        null
        1.08006
        1.07905
        1.07531
        1.08009
        1.08281
    ]
    """
    parent: Project | str | None

    tags: FxDailyTags

    version: int = 0
    type: Literal["FxDailyPx"] = "FxDailyPx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["FxDailyPxSnap"] = "FxDailyPxSnap"
 
    @classmethod
    def new(
        cls,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str = "raw",
        source: str | None = None,
        fx: str | None = None,
        parent: Project | str | None = None,
    ):
        # assert "USD" == symbol[3:], symbol
        # jpyusd = small, ie. the price in usd of jpy
        (
            ident,
            symbol,
            ord,
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
            expect=FX
        )
        assert adjust in FX_ADJUSTMENTS, adjust
        return cls(
            path=cls.build_path(
                parent,
                "fx-d",
                ident,
                strict=False,
            ),
            parent=parent,
            tags=FxDailyTags(
                symbol=symbol,
                ident=ord,
                field=field,
                adjust=adjust,
                fx=fx,
                source=source,
            ),
        )
    
    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, GraphSync):
            return HandleNewFxDailyPx()
        raise ValueError((self, event))

@dataclass(frozen=True)
class HandleUpdateFxDailyPx(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: FxDailyPx,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewFxDailyPx(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
    def handle(
        self,
        event: ObjectChange,
        snap: FxDailyPx.Snap,
        obj: FxDailyPx,
    ):
        tags = obj.tags

        if tags.source == "frd":
            df = apis.data.frd.read_file(
                os.environ.get("FRD_DATA", "TODO"),
                folder=apis.data.frd.Folders.FX,
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

        if "USD" == tags.symbol[3:]:
            pass
        else:
            df = df.with_columns(
                (1 / pl.col(tags.field))
                .alias(tags.field)
            )

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


@dataclass(frozen=True)
class DependsFxDailyPxParent(DependsParentValue):
    parent: FxDailyPx | str

    def get_parent(self) -> FxDailyPx:
        return context.get().get(self.parent, FxDailyPx)
    
#  ------------------

__all__ = [
    "FxDailyPx",
    "HandleUpdateFxDailyPx",
    "HandleNewFxDailyPx",
    "DependsFxDailyPxParent",
]
# TODO: if __name__ == __main__:
# run domain by name