
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from .m import FutDailyM, FutDailyMDomain, HandleNewFutDailyM

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyMD(FutDailyM):
    pass

@dataclass(frozen=True)
class FutDailyMDDomain(FutDailyMDomain):
    pass

@dataclass(frozen=True)
class HandleNewFutDailyMD(HandleNewFutDailyM):
    pass