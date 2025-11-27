
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
class FutDailyEVar(FutDailyEM):

    version: int = 0
    type: Literal["FutDailyEVar"] = "FutDailyEVar"

    SUFFIX: ClassVar[str] = "evar"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEVarSnap"] = "FutDailyEVarSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyEVar:
        return HandleNewFutDailyEVar()

@dataclass(frozen=True)
class HandleNewFutDailyEVar(HandleNewDailyEVar):
    pass


@dataclass(frozen=True)
class FutDailyEVarDomain(FutDailyDomain):
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
    >>> obj = FutDailyEVar.new(ord, span=4)
    >>> domain = (
    ...     FutDailyEVarDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-evar-4' [f64]
    [
        0.000066
        null
        null
        0.000124
        0.000083
        0.000139
        0.000084
        0.000154
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyEVarDomain"] = "FutDailyEVarDomain"


        SUFFIX: ClassVar[str] = "evar"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyEVar
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyEVar.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name