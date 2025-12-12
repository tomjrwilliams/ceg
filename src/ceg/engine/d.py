
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

from .. import apis
from ..apis import t
from ..apis import kv

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *
from .root import *


#  ------------------



@dataclass(frozen=True)
class DailyDomain(DomainWithEngineRoot):
    pass

# @dataclass(frozen=True)
# class DailyTags:
#     pass

@dataclass(frozen=True)
class DailyParams:
    pass

@dataclass(frozen=True)
class DailySnap(Snap):
    updated: dt.datetime
    first: dt.date # most recently added
    last: dt.date

@dataclass(frozen=True)
class Daily(
    # UseChunkedValues,
    DependsSelfNew,
    DependsDomainSync,
):
    parent: Object | str
    # tags: DailyTags
    # params: DailyParams

    # version: int = 0
    # type: Literal["DailyM"] = "DailyM"
    # NOTE: to change

    @dataclass(frozen=True) 
    class Snap(DailySnap):
        pass

    def get_snap(
        self,
        domain: Domain,
        t: Type[DailySnap] | None = None,
    ) -> DailySnap | None:
        s = self._get_snap(domain)
        if t is None:
            t = DailySnap
        if s is None:
            return s
        assert isinstance(s, t), (s, t)
        return s

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )
    
    def handler_object_new(
        self
    ) -> Handler: ...
    
    # TODO: handler object (self) change 
    # TODO: handler update target object

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(self.handler_object_new())

        # TODO: if upstream target change

        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class DailyChunked(
    Daily,
    UseChunkedValues,
):
    pass

def daily_to_df(
    obj: Daily,
    domain: Domain,
):
    snap = obj.get_snap(domain, t = DailySnap)
    assert snap is not None, (obj)
    v = (
        obj.value_key_val(domain)
        .get(obj.path, t=ValuePlSeries)
    )
    assert v is not None, obj
    return pl.DataFrame({
        v.obj: v.ser
    }).with_columns(
        pl.date_range(snap.first, snap.last).alias("date")
    )

__all__ = [
    "DailyDomain",
    # "DailyTags",
    "DailyParams",
    "Daily",
    "DailyChunked",
    "DailySnap",
    "daily_to_df",
]