
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

from .px import IndexDailyPx, HasIndexDailyPxParent

#  ------------------


@dataclass(frozen=True)
class IndexDailyRx(
    HasIndexDailyPxParent,
    BarDaily,
):
    parent: IndexDailyPx | str
    # TODO: or stock

    adjust: str

    version: int = 0
    type: Literal["IndexDailyRx"] = "IndexDailyRx"

    @dataclass(frozen=True)
    class Snap(IndexDaily.Snap):
        type: Literal["IndexDailyRxSnap"] = "IndexDailyRxSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> IndexDailyRx.Snap | None:
        return self.snap_key_val(domain).get(
            self.path, IndexDailyRx.Snap
        )
    
    @classmethod
    def new(
        cls,
        parent: IndexDailyPx | EngineRoot,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str = "abs",
        source: str | None = None,
        fx: str | None = None,
    ):
        if isinstance(parent, EngineRoot):
            assert isinstance(symbol, str), symbol
            if ident is not None:
                assert isinstance(ident, int), ident
            parent = IndexDailyPx.new(
                parent,
                symbol=symbol,
                ident=ident,
                field=field,
                adjust="raw",
                source=source,
                fx=fx,
            )
        assert adjust in RX_ADJUSTMENTS, adjust
        return cls(
            path=parent.extend_path("rx"),
            parent=parent,
            adjust=adjust,
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, DomainSync):
            res.append(HandleUpdateIndexDailyRx())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewIndexDailyRx())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateIndexDailyRx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: IndexDailyRx,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return

@dataclass(frozen=True)
class HandleNewIndexDailyRx(
    HandleNewDailyFromDailyParentPl
):
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: IndexDailyRx,
        domain: Domain,
    ) -> pl.DataFrame:
        
        parent = obj.get_parent(domain)
        
        e = (
            pl.col(VALUE).forward_fill().pct_change()
            if obj.adjust == "pct"
            else pl.col(VALUE).forward_fill().diff()
            if obj.adjust == "abs"
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
class IndexDailyDomain(IndexDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | IndexDaily
            | IndexDailyPx
            | IndexDailyRx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            IndexDailyPx.Snap
            | IndexDailyRx.Snap
        )

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

@dataclass(frozen=True)
class IndexDailyRxDomain(IndexDailyDomain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> ord = IndexDailyPx.new(
    ...     eng,
    ...     symbol="VIX",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd"
    ... )
    >>> obj = IndexDailyRx.new(ord, adjust="abs")
    >>> domain = (
    ...     IndexDailyRxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-idx-d-VIX-close-raw-frd-rx' [f64]
    [
        -0.52
        null
        null
        -1.8
        -0.33
        1.18
        0.36
        2.96
    ]
    """

    @dataclass(frozen=True)
    class Root(IndexDailyDomain.Root):
        parent: IndexDaily | str
        version: int = 0
        type: Literal["IndexDailyRxDomain"] = "IndexDailyRxDomain"

        SUFFIX: ClassVar[str] = "rx"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = IndexDailyRxDomain.Root.new(engine)
        return IndexDailyRxDomain(root)
    
    @classmethod
    def new(cls, symbol: str = "VIX") -> IndexDailyRxDomain:
        eng = EngineRoot.new(ROOT)
        self = IndexDailyRxDomain.empty(eng)
        for adj in RX_ADJUSTMENTS:
            obj = IndexDailyRx.new(
                eng,
                symbol=symbol,
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(IndexDailyRxDomain, self)
    
@dataclass(frozen=True)
class HasIndexDailyGrandParent(Object):
    parent: IndexDailyPx | IndexDailyRx | str

    def get_parent(self, domain: Domain) -> IndexDailyPx | IndexDailyRx:
        return domain.get(self.parent, IndexDailyPx | IndexDailyRx)

#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "IndexDailyRx",
    "HandleUpdateIndexDailyRx",
    "HandleNewIndexDailyRx",
    "IndexDailyDomain",
    "IndexDailyRxDomain",
    "HasIndexDailyGrandParent",
]