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
class IndexDailyTags(BarDailyTags):
    symbol: str
    ident: None
    field: str
    adjust: str
    source: str
    idx: str | None = None  # ccy to convert to


@dataclass(frozen=True)
class Index(UseRootConfigs, BarDaily):
    parent: EngineRoot | str
    version: int = 0
    type: Literal["Index"] = "Index"

    def get_parent(self, domain: Domain) -> EngineRoot:
        return domain.get(self.parent, EngineRoot)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain)
    
    @classmethod
    def new(cls, parent: EngineRoot, path = "idx"):
        return cls(
            path=f"{parent.path}-{path}",
            parent=parent,
        )
    
@dataclass(frozen=True)
class IndexDaily(UseRootConfigs):
    parent: Index | str
    version: int = 0
    type: Literal["IndexDaily"] = "IndexDaily"

    # TODO: fridx? for when downsample to week, month etc.

    @dataclass(frozen=True)
    class Snap(DailySnap):
        updated: dt.datetime
        first: dt.date # most recently added
        last: dt.date

    def get_parent(self, domain: Domain) -> Index:
        return domain.get(self.parent, Index)

    def root(self, domain: Domain) -> EngineRoot:
        return (
            self.get_parent(domain)
            .root(domain)
        )

    @classmethod
    def new(
        cls,
        parent: Index | EngineRoot,
        path = "d",
    ):
        if isinstance(parent, EngineRoot):
            parent = Index.new(parent)
        return cls(
            path=parent.extend_path(path),
            parent=parent,
        )
    
    
@dataclass(frozen=True)
class HasIndexDailyParent(Object):

    parent: IndexDaily | str

    def get_parent(self, domain: Domain) -> IndexDaily:
        return domain.get(self.parent, IndexDaily)


@dataclass(frozen=True)
class IndexDailyDomainBase(Domain):

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: IndexDaily | str

        SUFFIX: ClassVar[str] = "TODO"

        def get_parent(self, domain: Domain) -> IndexDaily:
            return domain.get(self.parent, IndexDaily)

        @classmethod
        def new(
            cls,
            parent: IndexDaily | EngineRoot,
        ):
            assert cls.SUFFIX != "TODO", cls
            if isinstance(parent, EngineRoot):
                parent = IndexDaily.new(parent)
            return cls(
                path = parent.extend_path(cls.SUFFIX),
                parent=parent
            )

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

IDX_ADJUSTMENTS = [
    # "full",
    "raw",
]

__all__ = [
    "Index",
    "IndexDaily",
    "IndexDailyTags",
    "IndexDailyDomainBase",
    "HasIndexDailyParent",
    "IDX_ADJUSTMENTS",
]