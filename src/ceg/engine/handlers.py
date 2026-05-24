from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Any,
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

from .. import apis
from ..apis import t, kv, logs, assert_is

from .objects import *
from .depends import *
from .values import *
from .events import *
from .snaps import *

from .daily import *

VALUE = "VALUE"

logger, log = logs.logger(
    __file__,
    streams=dict(formatter=True),
)


class HasSetFromDf(Protocol):
    def set_from_df(
        self,
        df: pl.DataFrame,
        k: str,
        obj: str,
        chunk: int | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class HandleNewValueFromPl(Handler):
    def set_value_from_df(
        self,
        obj: Object,
        df: pl.DataFrame,
        col: str,
        t: Type[HasSetFromDf],
    ):
        key_val = obj.value_key_val(t=t)
        return key_val.set_from_df(
            df,
            col,
            obj.id,
        )

    def key_value_type(
        self, obj: Object
    ) -> Type[HasSetFromDf]:
        # TODO: given parametrisation of object
        return ValueKeyChunkPlSeriesSQLite

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: Object,
    ) -> pl.Expr:
        raise ValueError(self)

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: Object,
    ) -> pl.DataFrame:
        # df = pl.DataFrame({VALUE: v.ser.fill_nan(None)})
        e = self.calc_expr(
            df,
            event,
            obj,
        )
        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )
        return df


@dataclass(frozen=True)
class DailyWithDailyParent(
    DependsParentValue,
    Daily,
):
    def get_parent(self) -> Daily:
        return context.get().get(self.parent, Daily)


@dataclass(frozen=True)
class HandleFromParent(Handler):
    pass


@dataclass(frozen=True)
class HandleFromParentPl(HandleFromParent):
    def get_parent_pl(
        self,
        event: ObjectChange,
        obj: Object,
        parent: Object | None = None,
        parent_snap: Snap | None = None,
    ):
        if parent is None:
            parent = assert_is(obj.get_parent())

        if parent_snap is None:
            parent_snap = assert_is(parent.get_snap())

        v = parent.value_key_val().get(parent.id)
        assert isinstance(v, ValuePlSeries), v

        return v


@dataclass(frozen=True)
class HandleNewValueFromParentPl(
    HandleNewValueFromPl,
    HandleFromParentPl,
):
    pass




@dataclass(frozen=True)
class HandleNewDailyFromDailyParent(Handler):
    def update_snap_from_parent(
        self,
        snap: Daily.Snap,
        obj: DailyWithDailyParent,
        parent: Daily | None = None,
        parent_snap: Daily.Snap | None = None,
    ):
        if parent_snap is None:
            if parent is None:
                parent = obj.get_parent()

            parent_snap = parent.get_snap()
            assert parent_snap is not None, parent_snap

        return replace(
            snap,
            first=parent_snap.first,
            last=parent_snap.last,
        )


@dataclass(frozen=True)
class HandleNewDailyFromDailyParentPl(
    HandleNewValueFromParentPl,
    HandleNewDailyFromDailyParent,
):
    def handle(
        self,
        event: ObjectChange,
        snap: Daily.Snap,
        obj: DailyWithDailyParent,
    ):
        # log("INFO")( handle_daily=obj.id)

        parent = obj.get_parent()
        parent_snap = parent.get_snap(Daily.Snap)

        if snap.first and snap.last:
            if (
                snap.first == parent_snap.first
                and snap.last == parent_snap.last
            ):
                return frozendict().set(
                    obj.id, (snap, [])
                )

        v = self.get_parent_pl(
            event,
            obj,
            parent=parent,
            parent_snap=parent_snap,
        )

        first = v.start or parent_snap.first
        last = v.end or parent_snap.last

        date_cols = (
            []
            if first is None or last is None
            else [pl.date_range(first, last).alias("date")]
        )

        df = v.to_df(
            alias=VALUE,
            pipe=lambda e: e.fill_nan(None),
            with_cols=date_cols,
        )
        assert "date" in df.schema, (
            df.schema,
            parent_snap,
            v,
        )

        df = df.pipe(self.calc_df, event, obj)

        v = self.set_value_from_df(
            obj,
            df,
            VALUE,
            t=self.key_value_type(obj),
        )

        snap = self.update_snap_from_parent(
            snap, obj, parent_snap=parent_snap
        )

        return ValueUpdate.from_snap(
            snap, into=frozendict()
        )


@dataclass(frozen=True)
class TriageUpstream(Object):
    def upstream_paths(self) -> list[str]: ...

    @dataclass(frozen=True)
    class Snap(Snap):
        triage: bool = True

        type: Literal["TriageUpstreamSnap"] = (
            "TriageUpstreamSnap"
        )


@dataclass(frozen=True)
class HandleUpdateLatestUpstream(Handler):
    def handle(
        self,
        event: ValueUpdate,
        snap: TriageUpstream.Snap,
        obj: TriageUpstream,
    ):
        es = [ValueUpdate(obj.id)]
        return frozendict().set(obj.id, (snap, es))


__all__ = [
    "VALUE",
    "DailyWithDailyParent",
    "HandleNewValueFromPl",
    "HandleFromParentPl",
    "HandleNewValueFromParentPl",
    "HandleNewDailyFromDf",
    "HandleNewDailyFromDailyParent",
    "HandleNewDailyFromDailyParentPl",
    "TriageUpstream",
    "HandleUpdateLatestUpstream",
]
