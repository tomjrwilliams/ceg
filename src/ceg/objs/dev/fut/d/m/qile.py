
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

@dataclass(frozen=True)
class FutDailyQile(FutDailyM):
    """FutDailyQile(UseRootConfigs):\n
    parent: FutDaily\n
    symbol: str\n
    field: str\n
    adjust: str\n
    fx: str | None = None # ccy to convert to\n
    source: str | None = None\n
    """
    parent: FutDailyOrdinal | FutDailyRx | str
    # TODO: or contract
    # TODO: or rx thereof

    symbol: str
    ident: str
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

    window: int | None = None
    quantile: float | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyQile"] = "FutDailyQile"

    SUFFIX: ClassVar[str] = "qile"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyQileSnap"] = "FutDailyQileSnap"
 
    def handler_object_new(
        self
    ) -> HandleNewFutDailyQile:
        return HandleNewFutDailyQile()
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        quantile: float | None = None,
    ):
        assert quantile is not None, cls
        if window is None:
            params = "ex"
        else:
            params = str(window)
        params += f"-{quantile}"
        return f"{cls.SUFFIX}-" + params
    
    @classmethod
    def new(
        cls,
        parent: FutDailyOrdinal | EngineRoot,
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
                window=window,quantile=quantile,
            )),
            parent=parent,
            symbol=parent.symbol,
            ident=str(ident),
            field=parent.field,
            adjust=parent.adjust,
            source=parent.source,
            fx=parent.fx,
            window=window,quantile=quantile,
        )

@dataclass(frozen=True)
class HandleNewFutDailyQile(HandleNewFutDailyM):

    def calc(
        self,
        event: Event,
        obj: FutDailyQile,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        assert obj.quantile is not None, obj
        
        e = (
            pl.col(obj.field).rolling_quantile(
                obj.quantile,
                window_size=len(df) if obj.window is None else obj.window,
                min_samples=1, # TODO: or more?
            )
        )

        df = df.with_columns(
            pl.when(pl.col(obj.field).is_null())
            .then(None)
            .otherwise(e)
            .alias(obj.field)
        )
        return df
    

@dataclass(frozen=True)
class FutDailyQileDomain(FutDailyMDomain):
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
    >>> obj = FutDailyQile.new(ord, window=4, quantile=.5)
    >>> domain = (
    ...     FutDailyQileDomain.empty(eng)
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
        FutDailyMDomain.Root,
    ):

        version: int = 0
        type: Literal["FutDailyQileDomain"] = "FutDailyQileDomain"

        # TODO: to override

        SUFFIX: ClassVar[str] = "qile"
        # TODO: to override

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
            | FutDailyQile
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            | FutDailyQile.Snap
        )

# TODO: if __name__ == __main__:
# run domain by name