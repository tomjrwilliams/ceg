from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Literal,
    ClassVar,
    overload,
)
import datetime as dt
import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

from ......apis import kv
from ......engine import *

from ..root import *


@dataclass(frozen=True)
class FutDailyOrdinalTags(BarDailyTags):
    symbol: str
    ident: int
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

@dataclass(frozen=True)
class FutDailyContractTags(BarDailyTags):
    symbol: str
    ident: str
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to



@dataclass(frozen=True)
class FutDaily(UseRootConfigs):
    parent: EngineRoot | str
    version: int = 0
    type: Literal["FutDaily"] = "FutDaily"

    # TODO: freq? for when downsample to week, month etc.

    @dataclass(frozen=True)
    class Snap(DailySnap):
        updated: dt.datetime
        first: dt.date # most recently added
        last: dt.date

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, EngineRoot)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain)

    @classmethod
    def new(
        cls,
        parent: EngineRoot,
        path = "fut-d",
    ):
        return cls(
            path=f"{parent.path}-{path}",
            parent=parent,
        )
    
    
@dataclass(frozen=True)
class HasFutDailyParent(Object):

    parent: FutDaily | str

    def get_parent(self, domain: Domain) -> FutDaily:
        return domain.get(self.parent, FutDaily)


@dataclass(frozen=True)
class FutDailyDomainBase(Domain):

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: FutDaily | str

        SUFFIX: ClassVar[str] = "TODO"

        def get_parent(self, domain: Domain) -> FutDaily:
            return domain.get(self.parent, FutDaily)

        @classmethod
        def new(
            cls,
            parent: FutDaily | EngineRoot,
        ):
            assert cls.SUFFIX != "TODO", cls
            if isinstance(parent, EngineRoot):
                parent = FutDaily.new(parent)
            return cls(
                path = parent.extend_path(cls.SUFFIX),
                parent=parent
            )
        

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

__all__ = [
    "FutDaily",
    "FutDailyOrdinalTags",
    "FutDailyContractTags",
    "FutDailyDomainBase",
    "HasFutDailyParent",
]