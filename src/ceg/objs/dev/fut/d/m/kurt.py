
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

from ...root import *
from ..bars import *
from ..calc import *

from ....calc import *

@dataclass(frozen=True)
class FutDailyKurt(FutDailyM):
    version: int = 0
    type: Literal["FutDailyKurt"] = "FutDailyKurt"

    SUFFIX: ClassVar[str] = "kurt"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyKurtSnap"] = "FutDailyKurtSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyKurt:
        return HandleNewFutDailyKurt()

@dataclass(frozen=True)
class HandleNewFutDailyKurt(HandleNewDailyKurt):
    pass

@dataclass(frozen=True)
class FutDailyKurtDomain(FutDailyDomain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> ord = FutDailyRx.new(
    ...     eng,
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = FutDailyKurt.new(ord, window=4)
    >>> domain = (
    ...     FutDailyKurtDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-kurt-4' [f64]
    [
        -1.116847
        null
        null
        -2.0
        -2.0
        -1.5
        -1.070286
        -1.379246
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyKurtDomain"] = "FutDailyKurtDomain"

        SUFFIX: ClassVar[str] = "kurt"
