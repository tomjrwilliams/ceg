
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
class FutDailyERms(FutDailyEM):

    version: int = 0
    type: Literal["FutDailyERms"] = "FutDailyERms"

    SUFFIX: ClassVar[str] = "erms"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyERmsSnap"] = "FutDailyERmsSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyERms:
        return HandleNewFutDailyERms()

@dataclass(frozen=True)
class HandleNewFutDailyERms(HandleNewDailyERms):
    pass


@dataclass(frozen=True)
class FutDailyERmsDomain(FutDailyDomain):
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
    >>> obj = FutDailyERms.new(ord, span=4)
    >>> domain = (
    ...     FutDailyERmsDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-erms-4' [f64]
    [
        0.007089
        null
        null
        0.012077
        0.009431
        0.010308
        0.008288
        0.014329
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyERmsDomain"] = "FutDailyERmsDomain"


        SUFFIX: ClassVar[str] = "erms"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyERms
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyERms.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name