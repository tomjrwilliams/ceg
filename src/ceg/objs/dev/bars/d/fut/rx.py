
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

from ..root import *
from .root import *
from .ord import FutDailyOrdinal, DependsFutDailyOrdinalParent, BarDaily

#  ------------------


@dataclass(frozen=True)
class FutDailyRx(
    DependsFutDailyOrdinalParent,
    BarDaily,
):
    parent: FutDailyOrdinal | str
    # TODO: or contract

    version: int = 0
    type: Literal["FutDailyRx"] = "FutDailyRx"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyRxSnap"] = "FutDailyRxSnap"
 
    def get_snap(
        self,
        domain: Domain,
        t: Type[DailySnap] | None = None,
    ) -> FutDailyRx.Snap | None:
        return self.snap_key_val(domain).get(
            self.path, FutDailyRx.Snap
        )
    
    @classmethod
    def new(
        cls,
        parent: FutDailyOrdinal | EngineRoot,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
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
        # assert ident is not None, parent
        return cls(
            path=parent.extend_path("rx"),
            parent=parent,
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_parent(domain)
        )

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, ValueAppend):
            res.append(HandleUpdateFutDailyRx())
        elif isinstance(event, ValueUpdate):
            assert self.get_parent(domain).path == event.obj, (self, event)
            res.append(HandleNewFutDailyRx())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateFutDailyRx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: FutDailyRx,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return

@dataclass(frozen=True)
class HandleNewFutDailyRx(
    HandleNewDailyFromDailyParentPl
):
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: FutDailyRx,
        domain: Domain,
    ) -> pl.DataFrame:
        
        parent = obj.get_parent(domain)

        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if parent.tags.adjust == "pct"
            else pl.col(VALUE).forward_fill().diff()
            if parent.tags.adjust == "abs"
            else None
        )
        assert e is not None, parent.tags.adjust

        df = df.with_columns(
            pl.when(pl.col(VALUE).is_null())
            .then(None)
            .otherwise(e)
            .alias(VALUE)
        )

        return df

# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------

@dataclass(frozen=True)
class FutDailyDomain(FutDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | FutDaily
            | FutDailyOrdinal
            | FutDailyRx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
        )

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

@dataclass(frozen=True)
class FutDailyRxDomain(FutDailyDomain):
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
    >>> obj = FutDailyRx.new(ord)
    >>> domain = (
    ...     FutDailyRxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd-rx' [f64]
    [
        0.000963
        null
        null
        0.017007
        0.001891
        -0.011499
        -0.003516
        -0.020255
    ]
    """

    @dataclass(frozen=True)
    class Root(FutDailyDomain.Root):
        parent: FutDaily | str
        version: int = 0
        type: Literal["FutDailyRxDomain"] = "FutDailyRxDomain"

        SUFFIX: ClassVar[str] = "rx"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = FutDailyRxDomain.Root.new(engine)
        return FutDailyRxDomain(root)
    
    @classmethod
    def new(cls, symbol: str = "ES") -> FutDailyRxDomain:
        eng = EngineRoot.new(ROOT)
        self = FutDailyRxDomain.empty(eng)
        for adj in ["pct", "abs"]:
            obj = FutDailyRx.new(
                eng,
                symbol=symbol, # spx
                ident = 1,
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(FutDailyRxDomain, self)
    
@dataclass(frozen=True)
class HasFutDailyGrandParent(Object):
    parent: FutDailyOrdinal | FutDailyRx | str

    def get_parent(self, domain: Domain) -> FutDailyOrdinal | FutDailyRx:
        return domain.get(self.parent, FutDailyOrdinal | FutDailyRx)

#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "FutDailyRx",
    "HandleUpdateFutDailyRx",
    "HandleNewFutDailyRx",
    "FutDailyDomain",
    "FutDailyRxDomain",
    "HasFutDailyGrandParent",
]