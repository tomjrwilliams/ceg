
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
class FutDailyMid(FutDailyM):
    version: int = 0
    type: Literal["FutDailyMid"] = "FutDailyMid"

    SUFFIX: ClassVar[str] = "mid"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyMidSnap"] = "FutDailyMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyMid:
        return HandleNewFutDailyMid()

    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        width: float | None = None,
    ):
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params + (
            ""
            if width is None
            else f"-{width}"
        )
    
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
        window: int | None = None,
        width: float | None = None,
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
                window=window,
                width=width,
            )),
            parent=parent,
            params=DailyMidParams(
                window=window,
                width=width,
            ),
        )
    
@dataclass(frozen=True)
class HandleNewFutDailyMid(HandleNewDailyMid):
    pass

@dataclass(frozen=True)
class FutDailyMidDomain(FutDailyDomain):
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
    >>> obj = FutDailyMid.new(ord, window=4)
    >>> domain = (
    ...     FutDailyMidDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-mid-4' [f64]
    [
        5699.5
        null
        null
        5766.875
        5821.0
        5793.0
        5782.875
        5724.75
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyMidDomain"] = "FutDailyMidDomain"


        SUFFIX: ClassVar[str] = "mid"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyMid
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyMid.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name