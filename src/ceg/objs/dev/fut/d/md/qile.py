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
class FutDailyDQile(FutDailyMD):
    version: int = 0
    type: Literal["FutDailyDQile"] = "FutDailyDQile"

    SUFFIX: ClassVar[str] = "dqile"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyDQileSnap"] = "FutDailyDQileSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyDQile:
        return HandleNewFutDailyDQile()
    
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
        quantile: float | None = None,
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
                quantile=quantile,
            )),
            parent=parent,
            params=DailyQileParams(
                window=window,
                quantile=quantile,
            ),
        )

@dataclass(frozen=True)
class HandleNewFutDailyDQile(HandleNewDailyDQile):
    pass
    

@dataclass(frozen=True)
class FutDailyDQileDomain(FutDailyDomain):
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
    >>> obj = FutDailyDQile.new(ord, window=4, quantile=.5)
    >>> domain = (
    ...     FutDailyDQileDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-dqile-4-0.5' [f64]
    [
        0.0
        null
        null
        0.0
        0.0
        -56.0
        -76.25
        -136.5
    ]
    """

    @dataclass(frozen=True)
    class Root(
        FutDailyDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyDQileDomain"] = "FutDailyDQileDomain"


        SUFFIX: ClassVar[str] = "dqile"

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyDQile
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyDQile.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name