
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
class FutDailySkew(FutDailyM):
    version: int = 0
    type: Literal["FutDailySkew"] = "FutDailySkew"

    SUFFIX: ClassVar[str] = "skew"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailySkewSnap"] = "FutDailySkewSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailySkew:
        return HandleNewFutDailySkew()

@dataclass(frozen=True)
class HandleNewFutDailySkew(HandleNewDailySkew):
    pass


@dataclass(frozen=True)
class FutDailySkewDomain(FutDailyDomain):
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
    >>> obj = FutDailySkew.new(ord, window=4)
    >>> domain = (
    ...     FutDailySkewDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx-skew-4' [f64]
    [
        0.15138
        null
        null
        2.6070e-14
        3.3046e-14
        0.073938
        0.464041
        -0.228863
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailySkewDomain"] = "FutDailySkewDomain"


        SUFFIX: ClassVar[str] = "skew"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailySkew
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailySkew.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name