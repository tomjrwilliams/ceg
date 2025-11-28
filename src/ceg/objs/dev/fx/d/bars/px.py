
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

from ....bars import BarDailyTags, BarDaily, FX
from ...root import *
from ....calc import *

#  ------------------


@dataclass(frozen=True)
class FxDailyPx(
    HasFxDailyParent,
    DailyChunked,
):
    parent: FxDaily | str

    tags: FxDailyTags

    version: int = 0
    type: Literal["FxDailyPx"] = "FxDailyPx"

    @dataclass(frozen=True)
    class Snap(FxDaily.Snap):
        type: Literal["FxDailyPxSnap"] = "FxDailyPxSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> FxDailyPx.Snap | None:
        return self.snap_key_val(domain).get(self.path, FxDailyPx.Snap)
    
    @classmethod
    def new(
        cls,
        parent: FxDaily | EngineRoot,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str = "raw",
        source: str | None = None,
        fx: str | None = None,
    ):
        if isinstance(parent, EngineRoot):
            parent = FxDaily.new(parent)
        (
            ident,
            symbol,
            ord,
            field,
            adjust,
            source,
            fx,
        ) = Fx.identifier(
            symbol, 
            ident, 
            field, 
            adjust, 
            source, 
            fx=fx,
            expect=FX
        )
        assert adjust in FX_ADJUSTMENTS, adjust
        return cls(
            path=parent.extend_path(ident),
            parent=parent,
            tags=FxDailyTags(
                symbol=symbol,
                ident=ord,
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
            res.append(HandleUpdateFxDailyPx())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewFxDailyPx())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateFxDailyPx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: FxDailyPx,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewFxDailyPx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: FxDailyPx,
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
                folder=apis.data.frd.Folders.FX,
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
            .set(obj.path, FxDailyPx.Snap(
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
class FxDailyDomainOnlyPx(FxDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fx
            | FxDaily
            | FxDaily
            | FxDailyPx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: FxDailyPx.Snap

@dataclass(frozen=True)
class FxDailyPxDomain(FxDailyDomainOnlyPx):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> obj = FxDailyPx.new(
    ...     eng,
    ...     symbol="EURUSD",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd",
    ... )
    >>> domain = (
    ...    FxDailyPxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fx-d-EURUSD-close-raw-frd' [f64]
    [
        1.08118
        null
        null
        1.08006
        1.07905
        1.07531
        1.08009
        1.08281
    ]
    """
    @dataclass(frozen=True)
    class Root(FxDailyDomainBase.Root):
        parent: FxDaily | str
        version: int = 0
        type: Literal["FxDailyPxDomain"] = "FxDailyPxDomain"

        SUFFIX: ClassVar[str] = "fx"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = FxDailyPxDomain.Root.new(engine)
        return FxDailyPxDomain(root)

    @classmethod
    def new(
        cls, symbol: str = "EURUSD"
    ) -> FxDailyPxDomain:
        eng = EngineRoot.new(ROOT)
        self = FxDailyPxDomain.empty(eng)
        for adj in IDX_ADJUSTMENTS:
            obj = FxDailyPx.new(
                eng,
                symbol=symbol, 
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(FxDailyPxDomain, self)
    
@dataclass(frozen=True)
class HasFxDailyPxParent(Object):
    parent: FxDailyPx | str

    def get_parent(self, domain: Domain) -> FxDailyPx:
        return domain.get(self.parent, FxDailyPx)

IDX_ADJUSTMENTS = ["raw"]

#  ------------------

__all__ = [
    "FxDailyPx",
    "HandleUpdateFxDailyPx",
    "HandleNewFxDailyPx",
    "FxDailyDomainOnlyPx",
    "FxDailyPxDomain",
    "HasFxDailyPxParent",
]
# TODO: if __name__ == __main__:
# run domain by name