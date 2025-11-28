
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

from ....bars import BarDailyTags, BarDaily, INDEX
from ...root import *

#  ------------------


@dataclass(frozen=True)
class IndexDailyPx(
    HasIndexDailyParent,
    DailyChunked,
):
    parent: IndexDaily | str

    tags: IndexDailyTags

    version: int = 0
    type: Literal["IndexDailyPx"] = "IndexDailyPx"

    @dataclass(frozen=True)
    class Snap(IndexDaily.Snap):
        type: Literal["IndexDailyPxSnap"] = "IndexDailyPxSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> IndexDailyPx.Snap | None:
        return self.snap_key_val(domain).get(self.path, IndexDailyPx.Snap)
    
    @classmethod
    def new(
        cls,
        parent: IndexDaily | EngineRoot,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str = "raw",
        source: str | None = None,
        fx: str | None = None,
    ):
        if isinstance(parent, EngineRoot):
            parent = IndexDaily.new(parent)
        (
            ident,
            symbol,
            idx,
            field,
            adjust,
            source,
            fx,
        ) = Index.identifier(
            symbol, 
            ident, 
            field, 
            adjust, 
            source, 
            fx=fx,
            expect=INDEX
        )
        assert adjust in IDX_ADJUSTMENTS, adjust
        return cls(
            path=parent.extend_path(ident),
            parent=parent,
            tags=IndexDailyTags(
                symbol=symbol,
                ident=idx,
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
            res.append(HandleUpdateIndexDailyPx())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewIndexDailyPx())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateIndexDailyPx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: IndexDailyPx,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewIndexDailyPx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: IndexDailyPx,
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
                folder=apis.data.frd.Folders.IND,
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
            .set(obj.path, IndexDailyPx.Snap(
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
class IndexDailyDomainOnlyPx(IndexDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Index
            | IndexDaily
            | IndexDaily
            | IndexDailyPx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: IndexDailyPx.Snap

@dataclass(frozen=True)
class IndexDailyPxDomain(IndexDailyDomainOnlyPx):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> obj = IndexDailyPx.new(
    ...     eng,
    ...     symbol="VIX",
    ...     field="close",
    ...     adjust="raw",
    ...     source="frd",
    ... )
    >>> domain = (
    ...    IndexDailyPxDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-idx-d-VIX-close-raw-frd' [f64]
    [
        19.28
        null
        null
        17.48
        17.15
        18.33
        18.69
        21.65
    ]
    """
    @dataclass(frozen=True)
    class Root(IndexDailyDomainBase.Root):
        parent: IndexDaily | str
        version: int = 0
        type: Literal["IndexDailyPxDomain"] = "IndexDailyPxDomain"

        SUFFIX: ClassVar[str] = "idx"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = IndexDailyPxDomain.Root.new(engine)
        return IndexDailyPxDomain(root)

    @classmethod
    def new(
        cls, symbol: str = "VIX"
    ) -> IndexDailyPxDomain:
        eng = EngineRoot.new(ROOT)
        self = IndexDailyPxDomain.empty(eng)
        for adj in IDX_ADJUSTMENTS:
            obj = IndexDailyPx.new(
                eng,
                symbol=symbol, 
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(IndexDailyPxDomain, self)
    
@dataclass(frozen=True)
class HasIndexDailyPxParent(Object):
    parent: IndexDailyPx | str

    def get_parent(self, domain: Domain) -> IndexDailyPx:
        return domain.get(self.parent, IndexDailyPx)

#  ------------------

__all__ = [
    "IndexDailyPx",
    "HandleUpdateIndexDailyPx",
    "HandleNewIndexDailyPx",
    "IndexDailyDomainOnlyPx",
    "IndexDailyPxDomain",
    "HasIndexDailyPxParent",
]
# TODO: if __name__ == __main__:
# run domain by name