
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

from ....calc import *

from ...root import *
from ..bars import *


#  ------------------

@dataclass(frozen=True)
class FutDailyEM(
    HasFutDailyGrandParent,
    DailyChunkedEM,
):
    parent: FutDailyOrdinal | FutDailyRx | str
    # TODO: or contract
    # TODO: or rx thereof

    params: DailyEMParams

    version: int = 0
    type: Literal["FutDailyEM"] = "FutDailyEM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "EM"
    # NOTE: to override
    
    @classmethod
    def new(
        cls,
        parent: FutDailyOrdinal | FutDailyRx | EngineRoot,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        # NOTE: just for contracts?
        source: str | None = None,
        fx: str | None = None,
        # TODO: expect, to decide on ordinal
        # --
        span: float | None = None,
        alpha: float | None = None,
    ):
        if isinstance(parent, EngineRoot):
            assert isinstance(symbol, str), symbol
            if ident is not None:
                assert isinstance(ident, int), ident
            parent = FutDailyOrdinal.new(
                parent,
                symbol=symbol,
                ident=ident,
                field=field,
                adjust=adjust,
                source=source,
                fx=fx,
            )
        return cls(
            path=parent.extend_path(cls.params_suffix(
                span=span,
                alpha=alpha,
            )),
            parent=parent,
            params=DailyEMParams(
                span=span,
                alpha=alpha,
            ),
        )
    
#  ------------------

__all__ = [
    "FutDailyEM"
]