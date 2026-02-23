
from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    overload,
    cast,
    final,
)
import datetime
import datetime as dt

import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import polars as pl

from ...... import apis
from ......apis import t
from ......apis import kv
from ......engine import *

from ..ident import *

#  ------------------

@dataclass(frozen=True)
class FutDailyContractTags(BarDailyTags):
    symbol: str
    ident: str
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to


# @dataclass(frozen=True)
# class FutDailyBarContractSnap(FutDailyBarSnap):
#     type: Literal["FutDailyBarContractSnap"] = "FutDailyBarContractSnap"

# @dataclass(frozen=True)
# class FutDailyBarContract(Object):
#     parent: FutDailyBar

#     symbol: str
#     field: str
#     adjust: str | None = None
#     expiry: str | None = None
#     fx: str | None = None  # ccy to convert to
#     source: str | None = None

#     version: int = 0
#     type: Literal["FutDailyBarContract"] = "FutDailyBarContract"

#     @classmethod
#     def new(
#         cls,
#         parent: FutDailyBar,
#         symbol: str,
#         expiry: str,
#         field: str,
#         adjust: str | None = None,
#         fx: str | None = None,
#         source: str | None = None,
#         path = "con",
#     ):
#         return cls(
#             path=f"{parent.path}-{path}",
#             parent=parent,
#             symbol=symbol,
#             expiry=expiry,
#             field=field,
#             adjust=adjust,
#             fx=fx,
#             source=source,
#         )

#     def value_key_val(self) -> ValueKeyVal:
#         # TODO: as for ordinal, but specific expiry
#         raise ValueError(self)

#     def depends(self) -> list[Dependency]:
#         # TODO: as for ordinal, but specific expiry
#         return []

#     def handler(self, event: Event, snap: Snap) -> Handler:
#         # TODO: as for ordinal, but specific expiry
#         return []

__all__ = [
    "FutDailyContractTags"
]