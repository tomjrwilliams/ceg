
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from .m import FutDailyM

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyMD(FutDailyM):
    pass

__all__ = ["FutDailyMD"]