
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
class IndexDailyTags(BarDailyTags):
    symbol: str
    ident: None
    field: str
    adjust: str
    source: str
    idx: str | None = None  # ccy to convert to

# VXTLT = tlt bond vol (from 2020 only)
# GVZ = gold
# EVZ eurocurrency
# OVX = crude vol
# vvix = vix vol

@dataclass(frozen=True)
class IndexDailyPx(
    DependsGraphSync,
    BarDaily,
):
    """
    >>> obj = IndexDailyPx.new(
    ...     symbol="VIX",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd",
    ... )
    >>> Graph().with_obj(obj).sync()
    >>> obj.value_key_val().get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-idx-d-VIX-close-raw-frd' [f64]
    [
        19.28
        null
        null
        17.48
        17.15
        18.33
        18.69
        21.65
    ]
    """
    parent: Project | str | None

    tags: IndexDailyTags

    version: int = 0
    type: Literal["IndexDailyPx"] = "IndexDailyPx"

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["IndexDailyPxSnap"] = "IndexDailyPxSnap"
 
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
        (
            ident,
            symbol,
            idx,
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
            expect=INDEX
        )
        assert adjust in IDX_ADJUSTMENTS, adjust
        return cls(
            path=cls.build_path(
                parent,
                "idx-d",
                ident,
                strict=False,
            ),
            parent=parent,
            tags=IndexDailyTags(
                symbol=symbol,
                ident=idx,
                field=field,
                adjust=adjust,
                fx=fx,
                source=source,
            ),
        )    

    def handler(self, event: Event, snap: Snap) -> Handler:
        if isinstance(event, GraphSync):
            return HandleNewIndexDailyPx()
        raise ValueError((self, event))

@dataclass(frozen=True)
class HandleUpdateIndexDailyPx(Handler):
    def handle(
        self,
        event: ObjectChange,
        obj: IndexDailyPx,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewIndexDailyPx(
    HandleNewValueFromPl,
    HandleNewDailyFromDf,
):
    def handle(
        self,
        event: ObjectChange,
        snap: IndexDailyPx.Snap,
        obj: IndexDailyPx,
    ):
        tags = obj.tags

        if tags.source == "frd":
            df = apis.data.frd.read_file(
                os.environ.get("FRD_DATA", "TODO"),
                folder=apis.data.frd.Folders.IND,
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
class DependsIndexDailyPxParent(DependsParentValue):
    parent: IndexDailyPx | str

    def get_parent(self) -> IndexDailyPx:
        return context.get().get(self.parent, IndexDailyPx)

#  ------------------

__all__ = [
    "IndexDailyPx",
    "HandleUpdateIndexDailyPx",
    "HandleNewIndexDailyPx",
    "DependsIndexDailyPxParent",
]
# TODO: if __name__ == __main__:
# run domain by name