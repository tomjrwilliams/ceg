
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from .......engine import *
from .m import BarDailyM

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class BarDailyMD(BarDailyM):
    pass

__all__ = ["BarDailyMD"]