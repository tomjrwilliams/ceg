
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

from .px import FxDailyPx, HasFxDailyPxParent

#  ------------------


@dataclass(frozen=True)
class FxDailyRx(
    HasFxDailyPxParent,
    BarDaily,
):
    parent: FxDailyPx | str
    # TODO: or stock

    adjust: str

    version: int = 0
    type: Literal["FxDailyRx"] = "FxDailyRx"

    @dataclass(frozen=True)
    class Snap(FxDaily.Snap):
        type: Literal["FxDailyRxSnap"] = "FxDailyRxSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> FxDailyRx.Snap | None:
        return self.snap_key_val(domain).get(
            self.path, FxDailyRx.Snap
        )
    
    @classmethod
    def new(
        cls,
        parent: FxDailyPx | EngineRoot,
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
            parent = FxDailyPx.new(
                parent,
                symbol=symbol,
                ident=ident,
                field=field,
                adjust="raw",
                source=source,
                fx=fx,
            )
        assert adjust in RX_ADJUSTMENTS, adjust
        # assert ident is not None, parent
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
            res.append(HandleUpdateFxDailyRx())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewFxDailyRx())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateFxDailyRx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: FxDailyRx,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, Wget and append
        return

@dataclass(frozen=True)
class HandleNewFxDailyRx(
    HandleNewDailyFromDailyParentPl
):
    
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectNew,
        obj: FxDailyRx,
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
class FxDailyDomain(FxDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | FxDaily
            | FxDailyPx
            | FxDailyRx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FxDailyPx.Snap
            | FxDailyRx.Snap
        )

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

@dataclass(frozen=True)
class FxDailyRxDomain(FxDailyDomain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> ord = FxDailyPx.new(
    ...     eng,
    ...     symbol="EURUSD",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd"
    ... )
    >>> obj = FxDailyRx.new(ord, adjust="abs")
    >>> domain = (
    ...     FxDailyRxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path).tail(8)
    shape: (8,)
    Series: 'test-fx-d-EURUSD-close-raw-frd-rx' [f64]
    [
        -0.00358
        null
        null
        -0.00112
        -0.00101
        -0.00374
        0.00478
        0.00272
    ]
    """

    @dataclass(frozen=True)
    class Root(FxDailyDomain.Root):
        parent: FxDaily | str
        version: int = 0
        type: Literal["FxDailyRxDomain"] = "FxDailyRxDomain"

        SUFFIX: ClassVar[str] = "rx"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = FxDailyRxDomain.Root.new(engine)
        return FxDailyRxDomain(root)
    
    @classmethod
    def new(cls, symbol: str = "EURUSD") -> FxDailyRxDomain:
        eng = EngineRoot.new(ROOT)
        self = FxDailyRxDomain.empty(eng)
        for adj in RX_ADJUSTMENTS:
            obj = FxDailyRx.new(
                eng,
                symbol=symbol,
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(FxDailyRxDomain, self)
    
@dataclass(frozen=True)
class HasFxDailyGrandParent(Object):
    parent: FxDailyPx | FxDailyRx | str

    def get_parent(self, domain: Domain) -> FxDailyPx | FxDailyRx:
        return domain.get(self.parent, FxDailyPx | FxDailyRx)

#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

__all__ = [
    "FxDailyRx",
    "HandleUpdateFxDailyRx",
    "HandleNewFxDailyRx",
    "FxDailyDomain",
    "FxDailyRxDomain",
    "HasFxDailyGrandParent",
]