from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Literal,
)
import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

from ....apis import kv
from ....engine import *


@dataclass(frozen=True)
class Bar(Object):
    parent: EngineRoot
    version: int = 0
    type: Literal["Bar"] = "Bar"

    @classmethod
    def new(cls, parent: EngineRoot, path = "bar"):
        return cls(
            path=f"{parent.path}.{path}",
            parent=parent,
        )
