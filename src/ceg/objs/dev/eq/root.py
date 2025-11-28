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

from ....apis import kv
from ....engine import *

from ..bars.d import *


@dataclass(frozen=True)
class EqDailyEtfTags(BarDailyTags):
    symbol: str
    ident: None
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

@dataclass(frozen=True)
class EqDailyStockTags(BarDailyTags):
    symbol: str
    ident: None # tbc
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to



@dataclass(frozen=True)
class Eq(UseRootConfigs, BarDaily):
    parent: EngineRoot | str
    version: int = 0
    type: Literal["Eq"] = "Eq"

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, EngineRoot)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain)
    
    @classmethod
    def new(cls, parent: EngineRoot, path = "eq"):
        return cls(
            path=f"{parent.path}-{path}",
            parent=parent,
        )
    
@dataclass(frozen=True)
class EqDaily(UseRootConfigs):
    parent: Eq | str
    version: int = 0
    type: Literal["EqDaily"] = "EqDaily"

    # TODO: freq? for when downsample to week, month etc.

    @dataclass(frozen=True)
    class Snap(DailySnap):
        updated: dt.datetime
        first: dt.date # most recently added
        last: dt.date

    def get_parent(self, domain: Domain) -> Eq:
        return domain.get(self.parent, Eq)

    def root(self, domain: Domain) -> EngineRoot:
        return (
            self.get_parent(domain)
            .root(domain)
        )

    @classmethod
    def new(
        cls,
        parent: Eq | EngineRoot,
        path = "d",
    ):
        if isinstance(parent, EngineRoot):
            parent = Eq.new(parent)
        return cls(
            path=parent.extend_path(path),
            parent=parent,
        )
    
    
@dataclass(frozen=True)
class HasEqDailyParent(Object):

    parent: EqDaily | str

    def get_parent(self, domain: Domain) -> EqDaily:
        return domain.get(self.parent, EqDaily)


@dataclass(frozen=True)
class EqDailyDomainBase(Domain):

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: EqDaily | str

        SUFFIX: ClassVar[str] = "TODO"

        def get_parent(self, domain: Domain) -> EqDaily:
            return domain.get(self.parent, EqDaily)

        @classmethod
        def new(
            cls,
            parent: EqDaily | EngineRoot,
        ):
            assert cls.SUFFIX != "TODO", cls
            if isinstance(parent, EngineRoot):
                parent = EqDaily.new(parent)
            return cls(
                path = parent.extend_path(cls.SUFFIX),
                parent=parent
            )

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

EQ_ADJUSTMENTS = [
    "full",
    "split",
    "raw",
]

__all__ = [
    "Eq",
    "EqDaily",
    "EqDailyEtfTags",
    "EqDailyStockTags",
    "EqDailyDomainBase",
    "HasEqDailyParent",
    "EQ_ADJUSTMENTS",
]