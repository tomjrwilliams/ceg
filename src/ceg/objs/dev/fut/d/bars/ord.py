
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

TEST = "test"
TEST_DB = "./__local__/test.db"

#  ------------------


@dataclass(frozen=True)
class FutDailyOrdinal(
    UseChunkedValues,
    UseRootConfigs,
):
    """FutDailyOrdinal(UseRootConfigs):\n
    parent: FutDaily\n
    symbol: str\n
    field: str\n
    adjust: str\n
    fx: str | None = None # ccy to convert to\n
    source: str | None = None\n
    """
    parent: FutDaily | str

    symbol: str
    ident: int
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyOrdinal"] = "FutDailyOrdinal"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyOrdinalSnap"] = "FutDailyOrdinalSnap"
 
    def get_parent(self, domain: Domain) -> FutDaily:
        return domain.get(self.parent, FutDaily)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain).root(domain)
    
    def get_snap(
        self, domain: Domain
    ) -> FutDailyOrdinal.Snap | None:
        snap = self.snap_key_val(domain).get(self.path)
        if snap is None:
            return snap
        assert isinstance(
            snap, FutDailyOrdinal.Snap
        ), snap
        return snap
    
    @classmethod
    def new(
        cls,
        parent: FutDaily | EngineRoot,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        # NOTE: just for contracts?
        source: str | None = None,
        fx: str | None = None,
    ):
        if isinstance(parent, EngineRoot):
            parent = FutDaily.new(parent)
        (
            ident,
            symbol,
            ord,
            field,
            adjust,
            source,
            fx,
        ) = Fut.identifier(
            symbol, 
            ident, 
            field, 
            adjust, 
            source, 
            fx=fx,
            expect=ORD
        )
        assert ord == 1, ord # TODO: for now only first
        return cls(
            path=parent.extend_path(ident),
            parent=parent,
            symbol=symbol,
            ident=ord,
            field=field,
            adjust=adjust,
            # expiry=expiry, # NOTE: just for contracts?
            fx=fx,
            source=source,
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        # TODO: listen to ancestors?
        return [
            Dependency(
                self.path,
                self.path,
                events=("ObjectNew",)
            ),
            Dependency(
                self.path,
                domain.root.path,
                events=("DomainSync",)
            ),
            # TODO: self.ObjectChange
            # for chunk and backend type / loc
        ]

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, DomainSync):
            res.append(HandleUpdateFutDailyOrdinal())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewFutDailyOrdinal())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateFutDailyOrdinal(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: FutDailyOrdinal,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewFutDailyOrdinal(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: FutDailyOrdinal,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            raise ValueError((self, event))

        if obj.source == "frd":
            df = apis.data.frd.read_file(
                os.environ.get("FRD_DATA", "TODO"),
                folder=apis.data.frd.Folders.FUT,
                symbol=obj.symbol,
                suffix=obj.adjust,
                snap = "full",  # full
                freq = "1day",  # 1day
                #
                start = None,
                end = None,
            ).select("date", obj.field)
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
            obj.field,
            chunk=kv.BODY,
            publish=True,
        )

        snap = (
            obj.snap_key_val(domain)
            .set(obj.path, FutDailyOrdinal.Snap(
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
class FutDailyOrdinalDomain(Domain):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> obj = FutDailyOrdinal.new(
    ...     eng,
    ...     symbol="ES", # spx
    ...     ident = 1,
    ...     field="close",
    ...     adjust="pct",
    ...     source="frd",
    ... )
    >>> domain = (
    ...    FutDailyOrdinalDomain.new(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-fut-d-ES-1-close-pct-frd' [f64]
    [
        5718.25
        null
        null
        5815.5
        5826.5
        5759.5
        5739.25
        5623.0
    ]
    """
    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: FutDaily | str
        version: int = 0
        type: Literal["FutDailyOrdinalDomain"] = "FutDailyOrdinalDomain"

        def get_parent(self, domain: Domain) -> FutDaily:
            return domain.get(self.parent, FutDaily)

        def root(self, domain: Domain) -> EngineRoot:
            return self.get_parent(domain).root(domain)

        @classmethod
        def new(
            cls,
            parent: FutDaily | EngineRoot,
        ):
            if isinstance(parent, EngineRoot):
                parent = FutDaily.new(parent)
            return cls(
                path = parent.extend_path("ord"),
                parent=parent
            )

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Fut
            | FutDaily
            | FutDaily
            | FutDailyOrdinal
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: FutDailyOrdinal.Snap

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

    @classmethod
    def new(cls, engine: EngineRoot):
        root = FutDailyOrdinalDomain.Root.new(engine)
        return FutDailyOrdinalDomain(root)
    
#  ------------------

# TODO: if __name__ == __main__:
# run domain by name