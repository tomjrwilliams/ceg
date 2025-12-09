
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

from .etf import EqDailyEtf, HasEqDailyEtfParent
from .stock import EqDailyStock, HasEqDailyStockParent

#  ------------------


@dataclass(frozen=True)
class EqDailyRx(
    HasEqDailyEtfParent,
    DailyChunked,
    BarDaily,
):
    parent: EqDailyEtf | EqDailyStock | str
    # TODO: or stock

    version: int = 0
    type: Literal["EqDailyRx"] = "EqDailyRx"

    @dataclass(frozen=True)
    class Snap(EqDaily.Snap):
        type: Literal["EqDailyRxSnap"] = "EqDailyRxSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> EqDailyRx.Snap | None:
        return self.snap_key_val(domain).get(
            self.path, EqDailyRx.Snap
        )
    
    @classmethod
    def new(
        cls,
        parent: EqDailyEtf | EngineRoot,
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
            parent = EqDailyEtf.new(
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
            + self.depends_domain(domain)
        )

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, DomainSync):
            res.append(HandleUpdateEqDailyRx())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewEqDailyRx())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateEqDailyRx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: EqDailyRx,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return

@dataclass(frozen=True)
class HandleNewEqDailyRx(
    HandleNewChunkedGivenPlSeries,
    HandleNewDailyChunkedFromChunkedParent,
):
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: EqDailyRx,
        domain: Domain,
    ) -> pl.DataFrame:
        
        parent = obj.get_parent(domain)
        
        # split: adjust by split ratio
        # dividends: 1 - (Dividend Amount / Prior Day's Close Price)

        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if parent.tags.adjust in [
                "pct", "split"
            ]
            else pl.col(VALUE).forward_fill().diff()
            if parent.tags.adjust in [
                "raw", "price", "full"
            ]
            # TODO: split only, raw
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
class EqDailyDomain(EqDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | EqDaily
            | EqDailyEtf
            | EqDailyRx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            EqDailyEtf.Snap
            | EqDailyRx.Snap
        )

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

@dataclass(frozen=True)
class EqDailyRxDomain(EqDailyDomain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> ord = EqDailyEtf.new(
    ...     eng,
    ...     symbol="SPY",
    ...     field="close",
    ...     adjust="full",
    ...     source="frd"
    ... )
    >>> obj = EqDailyRx.new(ord)
    >>> domain = (
    ...     EqDailyRxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-eq-d-SPY-close-full-frd-rx' [f64]
    [
        9.375071
        6.945601
        null
        5.529652
        null
        null
        -4.650274
        1.877995
    ]
    """

    @dataclass(frozen=True)
    class Root(EqDailyDomain.Root):
        parent: EqDaily | str
        version: int = 0
        type: Literal["EqDailyRxDomain"] = "EqDailyRxDomain"

        SUFFIX: ClassVar[str] = "rx"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = EqDailyRxDomain.Root.new(engine)
        return EqDailyRxDomain(root)
    
    @classmethod
    def new(cls, symbol: str = "SPY") -> EqDailyRxDomain:
        eng = EngineRoot.new(ROOT)
        self = EqDailyRxDomain.empty(eng)
        for adj in ["full"]:
            obj = EqDailyRx.new(
                eng,
                symbol=symbol,
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(EqDailyRxDomain, self)
    
@dataclass(frozen=True)
class HasEqDailyGrandParent(Object):
    parent: EqDailyEtf | EqDailyRx | str

    def get_parent(self, domain: Domain) -> EqDailyEtf | EqDailyRx:
        return domain.get(self.parent, EqDailyEtf | EqDailyRx)

#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "EqDailyRx",
    "HandleUpdateEqDailyRx",
    "HandleNewEqDailyRx",
    "EqDailyDomain",
    "EqDailyRxDomain",
    "HasEqDailyGrandParent",
]