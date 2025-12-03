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
class CalDaily(UseRootConfigs):
    parent: EngineRoot | str
    version: int = 0
    type: Literal["EqDaily"] = "EqDaily"

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
        path = "eq-d",
    ):
        return cls(
            path=parent.extend_path(path),
            parent=parent,
        )