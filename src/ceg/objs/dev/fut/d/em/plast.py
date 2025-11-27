
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
class FutDailyEPLast(FutDailyEM):

    version: int = 0
    type: Literal["FutDailyEPLast"] = "FutDailyEPLast"

    SUFFIX: ClassVar[str] = "eplast"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEPLastSnap"] = "FutDailyEPLastSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEPLast:
        return HandleNewFutDailyEPLast()

@dataclass(frozen=True)
class HandleNewFutDailyEPLast(HandleNewDailyEPLast):
    pass


@dataclass(frozen=True)
class FutDailyEPLastDomain(FutDailyDomain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> ord = FutDailyOrdinal.new(
    ...     eng,
    ...     symbol="ES",
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd"
    ... )
    >>> obj = FutDailyEPLast.new(ord, span=4)
    >>> domain = (
    ...     FutDailyEPLastDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-eplast-4' [f64]
    [
        0.278662
        null
        null
        0.00008
        0.000032
        0.840017
        0.936007
        0.999937
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEPLastDomain"] = "FutDailyEPLastDomain"


        SUFFIX: ClassVar[str] = "eplast"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyEPLast
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEPLast.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name