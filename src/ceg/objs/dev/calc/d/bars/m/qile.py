
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

from ....... import apis
from .......apis import t
from .......apis import kv
from .......engine import *

from ..prelude import *
from ..base.m import *
from ...base.m import *
from ...handlers.m import *

@dataclass(frozen=True)
class BarDailyQile(BarDailyM):
    version: int = 0
    type: Literal["BarDailyQile"] = "BarDailyQile"

    SUFFIX: ClassVar[str] = "qile"

    @dataclass(frozen=True)
    class Snap(DailySnap):
        type: Literal["BarDailyQileSnap"] = "BarDailyQileSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewBarDailyQile:
        return HandleNewBarDailyQile()
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        quantile: float | None = None,
    ):
        assert quantile is not None, (cls, quantile)
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params + f"-{quantile}"
    
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
        quantile: float | None = None,
        #
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
                quantile=quantile,
            )),
            parent=parent,
            params=DailyQileParams(
                window=window,
                quantile=quantile,
            ),
        )

@dataclass(frozen=True)
class HandleNewBarDailyQile(HandleNewDailyQile):
    pass


@dataclass(frozen=True)
class BarDailyQileDomain(DailyChunkedDomain):
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
    >>> obj = BarDailyQile.new(ord, window=4, quantile=.5)
    >>> domain = (
    ...     BarDailyQileDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-qile-4-0.5' [f64]
    [
        5718.25
        null
        null
        5815.5
        5826.5
        5815.5
        5815.5
        5759.5
    ]
    """

    @dataclass(frozen=True)
    class Root(
        DailyChunkedDomain.Root,
    ):

        version: int = 0
        type: Literal["BarDailyQileDomain"] = "BarDailyQileDomain"

        SUFFIX: ClassVar[str] = "qile"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | TBar
            | BarDailyQile
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            TBarSnap
            | BarDailyQile.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name
__all__ = [
    "BarDailyQile",
    "HandleNewBarDailyQile",
    "BarDailyQileDomain",
]