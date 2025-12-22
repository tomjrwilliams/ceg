
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
from ..apis import kv

from .objects import *
from .depends import *
from .values import *
from .events import *
from .snaps import *

from .d import *

VALUE = "VALUE"

from ..apis import logs

logger, log_kw = logs.prepare_logger(__file__)

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
        domain: Domain,
        df: pl.DataFrame,
        col: str,
        t: Type[HasSetFromDf],
        publish: bool = True,
    ):
        key_val = obj.value_key_val(domain, t = t)

        # TODO: assert df has length

        # if isinstance(key_val, ValueKeyChunkPlSeriesSQLite):
        #     v = key_val.set_from_df(
        #         df, col, obj.path,
        #     )
        # elif isinstance(key_val, ValueKeyChunkNumPySQLite):

        v = key_val.set_from_df(
            df, col, obj.path,
        )
        
        if publish:
            e = cast(
                ValueUpdate,
                obj.publish(ValueUpdate(obj.path), domain)
            )
        
        return e, v

    def key_value_type(
        self, obj: Object, domain: Domain
    ) -> Type[HasSetFromDf]:
        # TODO: given parametrisation of object
        return ValueKeyChunkPlSeriesSQLite

    def calc_expr(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: Object,
        domain: Domain,
    ) -> pl.Expr:
        raise ValueError(self)

    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: Object,
        domain: Domain,
    ) -> pl.DataFrame:
        # df = pl.DataFrame({VALUE: v.ser.fill_nan(None)})
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
    
    
@dataclass(frozen=True)
class DailyWithDailyParent(
    DependsParentValue,
    Daily,
):
    def get_parent(
        self, domain: Domain
    ) -> Daily:
        return domain.get(self.parent, Daily)
    
    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_parent(domain)
        )
    
    def handler_object_new(self) -> Handler: ...

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []

        if isinstance(event, ValueUpdate):
            assert self.get_parent(domain).path == event.obj, (self, event)
            res.append(self.handler_object_new())

        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleFromParent(Handler):
    pass

@dataclass(frozen=True)
class HandleFromParentPl(
    HandleFromParent
):

    def get_parent_pl(
        self,
        event: ObjectNew,
        obj: Object,
        domain: Domain,
    ):
        
        parent = obj.get_parent(domain)
        assert parent is not None, obj

        parent_snap = parent.get_snap(domain)
        assert parent_snap is not None, parent_snap

        v = (
            parent.value_key_val(domain)
            .get(parent.path)
        )
        assert isinstance(v, ValuePlSeries), v

        return v
    
@dataclass(frozen=True)
class HandleNewValueFromParentPl(
    HandleNewValueFromPl,
    HandleFromParentPl,
):
    pass

@dataclass(frozen=True)
class HandleNewDailyFromDf(
    Handler
):

    def set_snap_from_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: Daily,
        domain: Domain,        
    ):
        
        d_min, d_max = df.pipe(
            apis.pl.select_items,
            (
                pl.col("date").min(),
                pl.col("date").max(),
            ),
            t=dt.date
        )

        return (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event,
                obj.path,
                dt.datetime.now(),
                d_min,
                d_max,
            ))
        )
    
@dataclass(frozen=True)
class HandleNewDailyFromDailyParent(
    Handler
):

    def set_snap_from_parent(
        self,
        event: ObjectNew,
        obj: DailyWithDailyParent,
        domain: Domain,        
    ):
        
        parent = obj.get_parent(domain)

        parent_snap = parent.get_snap(domain)
        assert parent_snap is not None, parent_snap

        return (
            obj.snap_key_val(domain)
            .set(obj.path, type(obj).Snap(
                event=event,
                obj=obj.path,
                updated=dt.datetime.now(),
                first=parent_snap.first,
                last=parent_snap.last,
            ))
        )

@dataclass(frozen=True)
class HandleNewDailyFromDailyParentPl(
    HandleNewValueFromParentPl,
    HandleNewDailyFromDailyParent,
):
    
    def handle(
        self,
        event: ObjectNew,
        obj: DailyWithDailyParent,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            return None
            # # NOTE: should already be handled?
            # raise ValueError((self, event))

        v = self.get_parent_pl(
            event, obj, domain
        )

        df = (
            pl.DataFrame({VALUE: v.ser.fill_nan(None)})
            .pipe(self.calc_df, event, obj, domain)
        )

        e, v = self.set_value_from_df(
            obj,
            domain,
            df,
            VALUE,
            publish=True,
            t = self.key_value_type(obj, domain),
        )

        snap = self.set_snap_from_parent(
            event, obj, domain
        )

        return e

@dataclass(frozen=True)
class TracksLatestUpstream(Object):

    def upstream_paths(self) -> list[str]: ...

    @dataclass(frozen=True)
    class Snap(Snap):
        latest: int
        n_upstream: int
        upstream: dict[str, int]

        type: Literal["TracksLatestUpstreamSnap"] = "TracksLatestUpstreamSnap"

@dataclass(frozen=True)
class HandleUpdateLatestUpstream(Handler):

    def handle(
        self,
        event: ValueUpdate,
        obj: TracksLatestUpstream,
        domain: Domain,
    ):
        i = event.i
        assert i is not None, event

        snap = obj.get_snap(domain)

        if snap is None:
            upstream = {}
            latest = -1 # to ensure changed_i > latest
            snap = type(obj).Snap(
                event,
                obj.path,
                n_upstream=len(obj.upstream_paths()),
                latest=latest,
                upstream=upstream,
            )
        else:
            assert isinstance(snap, TracksLatestUpstream.Snap), snap
            upstream = snap.upstream
            latest = snap.latest

        n_upstream = len(upstream)

        if i > upstream.get(event.obj, -1):
            upstream = {
                **upstream,
                **{event.obj: i}
            }
            snap = replace(
                snap, 
                event=event,
                upstream=upstream,
            )

        log_kw(event=event, snap=snap)

        e = None
        if len(upstream) == snap.n_upstream and (
            i > latest or len(upstream) > n_upstream
         ):
            snap = replace(snap, latest=i)
            e = cast(
                ValueUpdate,
                obj.publish(ValueUpdate(obj.path), domain)
            )

        obj.snap_key_val(domain).set(obj.path, snap)
        
        return e
    
__all__ = [
    "VALUE",
    "DailyWithDailyParent",
    "HandleNewValueFromPl",
    "HandleFromParentPl",
    "HandleNewValueFromParentPl",
    "HandleNewDailyFromDf",
    "HandleNewDailyFromDailyParent",
    "HandleNewDailyFromDailyParentPl",
    "TracksLatestUpstream",
    "HandleUpdateLatestUpstream",
]