
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from .em import FutDailyEM, FutDailyEMDomain, HandleNewFutDailyEM

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyEMD(FutDailyEM):
    pass

@dataclass(frozen=True)
class FutDailyEMDDomain(FutDailyEMDomain):
    pass

@dataclass(frozen=True)
class HandleNewFutDailyEMD(HandleNewFutDailyEM):
    pass