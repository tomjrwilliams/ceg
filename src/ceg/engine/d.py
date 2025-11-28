
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
from .handlers import *
from .root import *


#  ------------------



@dataclass(frozen=True)
class DailyChunkedDomain(DomainWithEngineRoot):
    pass

# @dataclass(frozen=True)
# class DailyTags:
#     pass

@dataclass(frozen=True)
class DailyParams:
    pass


@dataclass(frozen=True)
class DailyChunked(
    UseChunkedValues,
    DependsSelfNew,
    DependsDomainSync,
):
    parent: Object | str
    # tags: DailyTags
    # params: DailyParams

    # version: int = 0
    # type: Literal["DailyM"] = "DailyM"
    # NOTE: to change

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )
    
    def handler_object_new(
        self
    ) -> HandleNewDaily: ...
    
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
class HandleNewDaily( # suffix with from df etc.?
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,

):
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: TDailyChunkedWithChunkedParent,
        domain: Domain,
    ) -> pl.DataFrame: ...


__all__ = [
    "DailyChunkedDomain",
    # "DailyTags",
    "DailyParams",
    "DailyChunked",
    "HandleNewDaily",
]