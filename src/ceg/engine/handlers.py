
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

from ..apis import kv

from .objects import *
from .values import *
from .snaps import *

VALUE = "VALUE"

def set_value_chunk_from_df(
    obj: Object,
    domain: Domain,
    df: pl.DataFrame,
    col: str,
    chunk: int,
    publish: bool = True,
):
    key_val = obj.value_key_val(domain, t = (
        ValueKeyChunkPlSeriesSQLite
        | ValueKeyChunkNumPySQLite
    ))

    # TODO: assert df has length

    if isinstance(key_val, ValueKeyChunkPlSeriesSQLite):
        v = key_val.set_from_df(
            df, col, obj.path, chunk
        )
    elif isinstance(key_val, ValueKeyChunkNumPySQLite):
        v = key_val.set_from_df(
            df, col, obj.path, chunk
        )
    else:
        raise ValueError((obj, key_val))
    
    if publish:
        e = cast(
            ValueUpdate, obj.publish(ValueUpdate(obj.path), domain)
        )
    
    return e, v

@dataclass(frozen=True)
class DailySnap(Snap):
    updated: dt.datetime
    first: dt.date # most recently added
    last: dt.date

@dataclass(frozen=True)
class TDailyChunked(Object):

    @dataclass(frozen=True) 
    class Snap(DailySnap):
        pass

    def get_snap(
        self, domain: Domain
    ) -> DailySnap | None: ...

    def value_key_val(
        self, domain: Domain, t: None = None,
    ) -> kv.KeyChunk: ...

@dataclass(frozen=True)
class TDailyChunkedWithChunkedParent(TDailyChunked):

    def get_parent(
        self, domain: Domain
    ) -> TDailyChunked: ...

@dataclass(frozen=True)
class HandleNewDailyChunkedFromChunkedParent(Handler):

    def handle_new(
        self,
        v: ValuePlSeries,
        event: ObjectNew,
        obj: TDailyChunkedWithChunkedParent,
        domain: Domain,
    ) -> ValueUpdate:
        raise ValueError(self)

    def handle(
        self,
        event: ObjectNew,
        obj: TDailyChunkedWithChunkedParent,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None

        parent = obj.get_parent(domain)

        parent_snap = parent.get_snap(domain)
        assert parent_snap is not None, parent_snap

        v = (
            parent.value_key_val(domain)
            .get(parent.path, kv.BODY)
        )
        assert isinstance(v, ValuePlSeries), v

        e = self.handle_new(
            v,
            event,
            obj,
            domain,
        )

        # TODO: if numpy we assume dense, use snap to infer dates

        snap = (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                parent_snap.first,
                parent_snap.last,
            ))
        )

        return e

@dataclass(frozen=True)
class HandleNewChunkedGivenPlSeries(Handler):

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: TDailyChunkedWithChunkedParent,
        domain: Domain,
    ) -> pl.Expr:
        raise ValueError(self)

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: TDailyChunkedWithChunkedParent,
        domain: Domain,
    ) -> pl.DataFrame:
        e = self.calc_expr(
            df,
            event,
            obj,
            domain,
        )
        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
        return df

    def handle_new(
        self,
        v: ValuePlSeries,
        event: ObjectNew,
        obj: TDailyChunkedWithChunkedParent,
        domain: Domain,
    ) -> ValueUpdate:
        # TODO: or return ValueNew?

        df = self.calc_df(
            pl.DataFrame({VALUE: v.ser.fill_nan(None)}),
            event,
            obj,
            domain,
        )

        e, _ = set_value_chunk_from_df(
            obj,
            domain,
            df,
            VALUE,
            chunk=kv.BODY,
            publish=True,
        )

        return e


__all__ = [
    "VALUE",
    "set_value_chunk_from_df",
    "DailySnap",
    "TDailyChunked",
    "TDailyChunkedWithChunkedParent",
    "HandleNewDailyChunkedFromChunkedParent",
    "HandleNewChunkedGivenPlSeries",
]