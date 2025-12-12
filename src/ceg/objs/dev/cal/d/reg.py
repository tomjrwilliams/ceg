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

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

from .root import *

# TODO: how to get date range without referring to parents?

#  ------------------

@dataclass(frozen=True)
class CalDailyPattern:
    n: int
    freq: str
    then: CalDailyPattern | None = None

    def path(self):
        return f"{self.freq}{self.n}"

@dataclass(frozen=True)
class CalDailyRegular(
    Daily,
):
    parent: CalDaily | str

    pattern: CalDailyPattern
    
    version: int = 0
    type: Literal["CalDailyRegular"] = "CalDailyRegular"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["CalDailyRegularSnap"] = "CalDailyRegularSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> CalDailyRegular.Snap | None:
        return self.snap_key_val(domain).get(self.path, CalDailyRegular.Snap)
    
    @classmethod
    def new(
        cls,
        parent: CalDaily | EngineRoot,
        n: int,
        freq: str,
        # TODO: then
    ):
        if isinstance(parent, EngineRoot):
            parent = CalDaily.new(parent)
        pattern=CalDailyPattern(
            n=n,
            freq=freq
        )
        return cls(
            path=parent.extend_path(pattern.path()),
            parent=parent,
            pattern=pattern
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, DomainSync):
            res.append(HandleUpdateCalDailyRegular())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewCalDailyRegular())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateCalDailyRegular(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: CalDailyRegular,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewCalDailyRegular(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: CalDailyRegular,
        domain: Domain,
    ):
        pattern = obj.pattern
        snap = obj.get_snap(domain)

        if snap is not None:
            return None
            # # NOTE: should already be handled?
            # raise ValueError((self, event))
    
        raise ValueError(self)
        d_min, d_max = None

        e, v = handlers.set_value_from_df(
            obj,
            domain,
            df,
            "cal",
            chunk=kv.BODY,
            publish=True,
        )

        snap = (
            obj.snap_key_val(domain)
            .set(obj.path, CalDailyRegular.Snap(
                event,
                obj.path,
                dt.datetime.now(),
                d_min,
                d_max,
            ))
        )

        return e

# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------

@dataclass(frozen=True)
class CalDailyDomainOnlyEtf(Domain):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | CalDaily
            | CalDailyRegular
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: CalDailyRegular.Snap


# @dataclass(frozen=True)
# class CalDailyRegularDomain(CalDailyDomainOnlyEtf):
#     """
#     >>> eng = EngineRoot.new(ROOT)
#     >>> obj = CalDailyRegular.new(
#     ...     eng,
#     ...     symbol="SPY",
#     ...     field="close",
#     ...     adjust="full",
#     ...     source="frd",
#     ... )
#     >>> domain = (
#     ...    CalDailyRegularDomain.empty(eng)
#     ...     .with_obj(obj)
#     ...     .sync()
#     ... )
#     >>> obj.value_key_val(domain).get(obj.path).tail(8)
#     shape: (8,)
#     Series: 'test-eq-d-SPY-close-full-frd' [f64]
#     [
#         563.98
#         null
#         null
#         574.08
#         575.46
#         568.59
#         567.08
#         555.66
#     ]
#     """
#     @dataclass(frozen=True)
#     class Root(CalDailyDomainBase.Root):
#         parent: CalDaily | str
#         version: int = 0
#         type: Literal["CalDailyRegularDomain"] = "CalDailyRegularDomain"

#         SUFFIX: ClassVar[str] = "etf"

#     @classmethod
#     def empty(cls, engine: EngineRoot):
#         root = CalDailyRegularDomain.Root.new(engine)
#         return CalDailyRegularDomain(root)

#     @classmethod
#     def new(
#         cls, symbol: str = "SPY"
#     ) -> CalDailyRegularDomain:
#         eng = EngineRoot.new(ROOT)
#         self = CalDailyRegularDomain.empty(eng)
#         for adj in EQ_ADJUSTMENTS:
#             obj = CalDailyRegular.new(
#                 eng,
#                 symbol=symbol, 
#                 ident = 1,
#                 field="close",
#                 adjust=adj,
#                 source="frd",
#             )
#             self = self.with_obj(obj)
#         return cast(CalDailyRegularDomain, self)
    
# @dataclass(frozen=True)
# class HasCalDailyRegularParent(Object):
#     parent: CalDailyRegular | str

#     def get_parent(self, domain: Domain) -> CalDailyRegular:
#         return domain.get(self.parent, CalDailyRegular)

#  ------------------

__all__ = [
    "CalDailyRegular",
    "HandleUpdateCalDailyRegular",
    "HandleNewCalDailyRegular",
    "CalDailyDomainOnlyEtf",
    # "CalDailyRegularDomain",
    # "HasCalDailyRegularParent",
]
# TODO: if __name__ == __main__:
# run domain by name
