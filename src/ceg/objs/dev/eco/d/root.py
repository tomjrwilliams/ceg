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

from .....apis import kv
from .....engine import *


@dataclass(frozen=True)
class EcoDailyTags:
    symbol: str
    source: str
    units: str
    frequency: str
    seasonal: str

    asof: str | None = None
    lag: dt.timedelta | None = None

    transform: str | None = None

    # etc.

    
@dataclass(frozen=True)
class EcoDaily(UseRootConfigs):
    parent: EngineRoot | str
    version: int = 0
    type: Literal["EcoDaily"] = "EcoDaily"

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
        path = "eco-d",
    ):
        return cls(
            path=parent.extend_path(path),
            parent=parent,
        )
    
    
@dataclass(frozen=True)
class HasEcoDailyParent(Object):

    parent: EcoDaily | str

    def get_parent(self, domain: Domain) -> EcoDaily:
        return domain.get(self.parent, EcoDaily)


@dataclass(frozen=True)
class EcoDailyDomainBase(Domain):

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: EcoDaily | str

        SUFFIX: ClassVar[str] = "TODO"

        def get_parent(self, domain: Domain) -> EcoDaily:
            return domain.get(self.parent, EcoDaily)

        @classmethod
        def new(
            cls,
            parent: EcoDaily | EngineRoot,
        ):
            assert cls.SUFFIX != "TODO", cls
            if isinstance(parent, EngineRoot):
                parent = EcoDaily.new(parent)
            return cls(
                path = parent.extend_path(cls.SUFFIX),
                parent=parent
            )

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

__all__ = [
    "EcoDaily",
    "EcoDailyTags",
    "EcoDailyDomainBase",
    "HasEcoDailyParent",
]