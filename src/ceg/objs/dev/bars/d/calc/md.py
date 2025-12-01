
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from ...... import apis
from ......apis import t
from ......apis import kv
from ......engine import *

from ..root import *
from ....calc.d import *
from .m import BarDailyM

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class BarDailyMD(BarDailyM):
    pass

__all__ = ["BarDailyMD"]