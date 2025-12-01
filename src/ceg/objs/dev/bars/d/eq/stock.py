
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

#  ------------------


@dataclass(frozen=True)
class EqDailyStock(
    HasEqDailyParent,
    DailyChunked,
    BarDaily,
):
    parent: EqDaily | str

    tags: EqDailyStockTags
    
    version: int = 0
    type: Literal["EqDailyStock"] = "EqDailyStock"

    @dataclass(frozen=True)
    class Snap(EqDaily.Snap):
        type: Literal["EqDailyStockSnap"] = "EqDailyStockSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> EqDailyStock.Snap | None:
        return self.snap_key_val(domain).get(self.path, EqDailyStock.Snap)
    
    @classmethod
    def new(
        cls,
        parent: EqDaily | EngineRoot,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
    ):
        if isinstance(parent, EngineRoot):
            parent = EqDaily.new(parent)
        (
            ident,
            symbol,
            stock,
            field,
            adjust,
            source,
            fx,
        ) = cls.identifier(
            symbol, 
            ident, 
            field, 
            adjust, 
            source, 
            fx=fx,
            expect=STOCK
        )
        assert adjust in EQ_ADJUSTMENTS, adjust
        return cls(
            path=parent.extend_path(ident),
            parent=parent,
            tags=EqDailyStockTags(
                symbol=symbol,
                ident=stock,
                field=field,
                adjust=adjust,
                fx=fx,
                source=source,
            ),
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, DomainSync):
            res.append(HandleUpdateEqDailyStock())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewEqDailyStock())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateEqDailyStock(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: EqDailyStock,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewEqDailyStock(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: EqDailyStock,
        domain: Domain,
    ):
        tags = obj.tags
        snap = obj.get_snap(domain)

        if snap is not None:
            return None
            # # NOTE: should already be handled?
            # raise ValueError((self, event))

        if tags.source == "frd":
            df = apis.data.frd.read_file(
                os.environ.get("FRD_DATA", "TODO"),
                folder=apis.data.frd.Folders.STOCK,
                symbol=tags.symbol,
                suffix=tags.adjust,
                snap = "full",  # full
                freq = "1day",  # 1day
                #
                start = None,
                end = None,
            ).select("date", tags.field)
        else:
            raise ValueError(obj)

        d_min, d_max = df.pipe(
            apis.pl.select_items,
            (
                pl.col("date").min(),
                pl.col("date").max(),
            ),
            t=dt.date
        )

        e, v = handlers.set_value_chunk_from_df(
            obj,
            domain,
            df,
            tags.field,
            chunk=kv.BODY,
            publish=True,
        )

        snap = (
            obj.snap_key_val(domain)
            .set(obj.path, EqDailyStock.Snap(
                event,
                obj.path,
                dt.datetime.now(),
                d_min,
                d_max,
            ))
        )

        return e

# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------

@dataclass(frozen=True)
class EqDailyDomainOnlyStock(EqDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | EqDaily
            | EqDailyStock
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: EqDailyStock.Snap

# TODO: stock data not yet downloaded

@dataclass(frozen=True)
class EqDailyStockDomain(EqDailyDomainOnlyStock):
    # """
    # >>> eng = EngineRoot.new(ROOT)
    # >>> obj = EqDailyStock.new(
    # ...     eng,
    # ...     symbol="AAPL",
    # ...     field="close",
    # ...     adjust="full",
    # ...     source="frd",
    # ... )
    # >>> domain = (
    # ...    EqDailyStockDomain.empty(eng)
    # ...     .with_obj(obj)
    # ...     .sync()
    # ... )
    # >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    # shape: (8,)
    # Series: 'test-fut-d-AAPL-1-close-full-frd' [f64]
    # [
    #     5718.25
    #     null
    #     null
    #     5815.5
    #     5826.5
    #     5759.5
    #     5739.25
    #     5623.0
    # ]
    # """
    @dataclass(frozen=True)
    class Root(EqDailyDomainBase.Root):
        parent: EqDaily | str
        version: int = 0
        type: Literal["EqDailyStockDomain"] = "EqDailyStockDomain"

        SUFFIX: ClassVar[str] = "stock"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = EqDailyStockDomain.Root.new(engine)
        return EqDailyStockDomain(root)

    @classmethod
    def new(
        cls, symbol: str = "AAPL"
    ) -> EqDailyStockDomain:
        eng = EngineRoot.new(ROOT)
        self = EqDailyStockDomain.empty(eng)
        for adj in EQ_ADJUSTMENTS:
            obj = EqDailyStock.new(
                eng,
                symbol=symbol, 
                ident = 1,
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(EqDailyStockDomain, self)
    
@dataclass(frozen=True)
class HasEqDailyStockParent(Object):
    parent: EqDailyStock | str

    def get_parent(self, domain: Domain) -> EqDailyStock:
        return domain.get(self.parent, EqDailyStock)

#  ------------------

__all__ = [
    "EqDailyStock",
    "HandleUpdateEqDailyStock",
    "HandleNewEqDailyStock",
    "EqDailyDomainOnlyStock",
    "EqDailyStockDomain",
    "HasEqDailyStockParent",
]
# TODO: if __name__ == __main__:
# run domain by name