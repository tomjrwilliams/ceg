
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from .......engine import *
from .em import BarDailyEM

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class BarDailyDEM(BarDailyEM):
    pass

__all__ = ["BarDailyDEM"]