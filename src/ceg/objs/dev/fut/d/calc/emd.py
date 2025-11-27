
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from .em import FutDailyEM

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyDEM(FutDailyEM):
    pass

__all__ = ["FutDailyDEM"]