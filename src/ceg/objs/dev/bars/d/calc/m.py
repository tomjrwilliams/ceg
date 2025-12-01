
from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    Any,
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

from ..root import *
from ....calc.d import *
from ..prelude import *


#  ------------------

@dataclass(frozen=True)
class BarDailyM(
    DailyChunkedM,
):
    parent: (
        str
        | FutDailyOrdinal
        | FutDailyRx
        | EqDailyEtf
        | EqDailyStock
        | EqDailyRx
        | IndexDailyPx
        | IndexDailyRx
        | FxDailyPx
        | FxDailyRx
    )
    
    params: DailyMParams

    version: int = 0
    type: Literal["BarDailyM"] = "BarDailyM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "M"
    # NOTE: to override
    
    @classmethod
    def new(
        cls,
        parent: (
            EngineRoot
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyStock
            | EqDailyRx
            | IndexDailyPx
            | IndexDailyRx
            | FxDailyPx
            | FxDailyRx
        ),
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        # --
        window: int | None = None,
        t: (
            None
            | Type[FutDailyOrdinal]
            | Type[FutDailyRx]
            | Type[EqDailyEtf]
            | Type[EqDailyStock]
            | Type[EqDailyRx]
            | Type[IndexDailyPx]
            | Type[IndexDailyRx]
            | Type[FxDailyPx]
            | Type[FxDailyRx]
        ) = None,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            assert isinstance(symbol, str), (cls, t)
            assert adjust is not None, adjust
            parent = t.new(
                parent,
                symbol,
                ident,
                field,
                adjust,
                source,
                fx,
            )
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=window,
            )),
            parent=parent,
            params=DailyMParams(window=window),
        )
    
#  ------------------

__all__ = ["BarDailyM"]