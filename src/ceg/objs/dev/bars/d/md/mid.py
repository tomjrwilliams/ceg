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

from ..prelude import *
from ..calc.md import *
from ....calc.d import *

@dataclass(frozen=True)
class BarDailyDMid(BarDailyMD):
    version: int = 0
    type: Literal["BarDailyDMid"] = "BarDailyDMid"

    SUFFIX: ClassVar[str] = "dmid"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyDMidSnap"] = "BarDailyDMidSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyDMid:
        return HandleNewBarDailyDMid()
    
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
        width: float | None = None,
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
class HandleNewBarDailyDMid(HandleNewDailyDMid):
    pass
    

@dataclass(frozen=True)
class BarDailyDMidDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyDMid.new(ord, window=4)
    >>> domain = (
    ...     BarDailyDMidDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dmid-4' [f64]
    [
        18.75
        null
        null
        48.625
        5.5
        -33.5
        -43.625
        -101.75
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyDMidDomain"] = "BarDailyDMidDomain"

        SUFFIX: ClassVar[str] = "dmid"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyDMid
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyDMid.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyDMid",
    "HandleNewBarDailyDMid",
    "BarDailyDMidDomain",
]