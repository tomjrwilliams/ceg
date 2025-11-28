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

from ..bars import BarDailyTags, BarDaily


@dataclass(frozen=True)
class FxDailyTags(BarDailyTags):
    symbol: str
    ident: None
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to


@dataclass(frozen=True)
class Fx(UseRootConfigs, BarDaily):
    parent: EngineRoot | str
    version: int = 0
    type: Literal["Fx"] = "Fx"

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, EngineRoot)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain)
    
    @classmethod
    def new(cls, parent: EngineRoot, path = "fx"):
        return cls(
            path=f"{parent.path}-{path}",
            parent=parent,
        )
    
@dataclass(frozen=True)
class FxDaily(UseRootConfigs):
    parent: Fx | str
    version: int = 0
    type: Literal["FxDaily"] = "FxDaily"

    # TODO: frfx? for when downsample to week, month etc.

    @dataclass(frozen=True)
    class Snap(DailySnap):
        updated: dt.datetime
        first: dt.date # most recently added
        last: dt.date

    def get_parent(self, domain: Domain) -> Fx:
        return domain.get(self.parent, Fx)

    def root(self, domain: Domain) -> EngineRoot:
        return (
            self.get_parent(domain)
            .root(domain)
        )

    @classmethod
    def new(
        cls,
        parent: Fx | EngineRoot,
        path = "d",
    ):
        if isinstance(parent, EngineRoot):
            parent = Fx.new(parent)
        return cls(
            path=parent.extend_path(path),
            parent=parent,
        )
    
    
@dataclass(frozen=True)
class HasFxDailyParent(Object):

    parent: FxDaily | str

    def get_parent(self, domain: Domain) -> FxDaily:
        return domain.get(self.parent, FxDaily)


@dataclass(frozen=True)
class FxDailyDomainBase(Domain):

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: FxDaily | str

        SUFFIX: ClassVar[str] = "TODO"

        def get_parent(self, domain: Domain) -> FxDaily:
            return domain.get(self.parent, FxDaily)

        @classmethod
        def new(
            cls,
            parent: FxDaily | EngineRoot,
        ):
            assert cls.SUFFIX != "TODO", cls
            if isinstance(parent, EngineRoot):
                parent = FxDaily.new(parent)
            return cls(
                path = parent.extend_path(cls.SUFFIX),
                parent=parent
            )

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

FX_ADJUSTMENTS = [
    # "full",
    "raw",
]

__all__ = [
    "Fx",
    "FxDaily",
    "FxDailyTags",
    "FxDailyDomainBase",
    "HasFxDailyParent",
    "FX_ADJUSTMENTS",
]