
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

from ..... import apis
from .....apis import type_vars
from .....apis import kv
from .....engine import *

from ..root import Fut, FutDaily, ORD, CONTR
from .bars.ord import FutDailyOrdinal

TEST = "test"
TEST_DB = "./__local__/test.db"

#  ------------------


@dataclass(frozen=True)
class FutDailyRx(
    UseChunkedValues,
    UseRootConfigs,
):
    """FutDailyRx(UseRootConfigs):\n
    parent: FutDaily\n
    symbol: str\n
    field: str\n
    adjust: str\n
    fx: str | None = None # ccy to convert to\n
    source: str | None = None\n
    """
    parent: FutDailyOrdinal | str
    # TODO: or contract

    symbol: str
    ident: str
    field: str
    adjust: str
    fx: str | None = None  # ccy to convert to
    source: str | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyRx"] = "FutDailyRx"

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyRxSnap"] = "FutDailyRxSnap"
 
    def get_parent(self, domain: Domain) -> FutDailyOrdinal:
        return domain.get(self.parent, FutDailyOrdinal)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain).root(domain)
    
    def get_snap(
        self, domain: Domain
    ) -> FutDailyRx.Snap | None:
        snap = self.snap_key_val(domain).get(self.path)
        if snap is None:
            return snap
        assert isinstance(
            snap, FutDailyRx.Snap
        ), snap
        return snap
    
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
            path=parent.extend_path("rx"),
            parent=parent,
            symbol=parent.symbol,
            ident=str(ident),
            field=parent.field,
            adjust=parent.adjust,
            source=parent.source,
            fx=parent.fx,
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
            res.append(HandleUpdateFutDailyRx())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
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
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewFutDailyRx(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: FutDailyRx,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            raise ValueError((self, event))

        parent = obj.get_parent(domain)

        parent_snap = parent.get_snap(domain)
        assert parent_snap is not None, parent_snap

        v = (
            parent.value_key_val(domain)
            .get(parent.path, kv.BODY)
        )

        # TODO: if numpy we assume dense, use snap to infer dates

        assert isinstance(v, ValuePlSeries), v
        df = pl.DataFrame({obj.field: v.ser})

        e = (
            pl.col(obj.field).pct_change()
            if parent.adjust == "pct"
            else pl.col(obj.field).forward_fill().diff()
            if parent.adjust == "abs"
            else None
        )
        assert e is not None, parent.adjust

        df = df.with_columns(
            pl.when(pl.col(obj.field).is_null())
            .then(None)
            .otherwise(e)
            .alias(obj.field)
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
            .set(obj.path, FutDailyRx.Snap(
                event,
                obj.path,
                dt.datetime.now(),
                parent_snap.first,
                parent_snap.last,
            ))
        )

        return e

# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------

@dataclass(frozen=True)
class FutDailyRxDomain(Domain):
    """
    >>> from .bars import FutDailyOrdinalDomain
    >>> eng = EngineRoot.new(TEST)
    >>> ord = FutDailyOrdinal.new(
    ...     eng,
    ...     symbol="TN", # 10yr us
    ...     ident = 1,
    ...     field="close",
    ...     adjust="abs",
    ...     source="frd"
    ... )
    >>> obj = FutDailyRx.new(ord)
    >>> domain = (
    ...    FutDailyOrdinalDomain.new(eng)
    ...     .with_obj(ord)
    ...     .sync()
    ... )
    >>> domain = (
    ...     FutDailyRxDomain.new(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> ord.value_key_val(domain).get(ord.path, 1).tail(8)
    shape: (8,)
    Series: 'test.fut.d.TN.1.close.abs.frd' [f64]
    [
        114.0625
        null
        null
        113.359375
        113.578125
        113.28125
        113.015625
        114.03125
    ]
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test.fut.d.TN.1.close.abs.frd.rx' [f64]
    [
        -0.171875
        null
        null
        -0.703125
        0.21875
        -0.296875
        -0.265625
        1.015625
    ]
    """

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: FutDaily | str
        version: int = 0
        type: Literal["FutDailyRxDomain"] = "FutDailyRxDomain"

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
                path = parent.extend_path("rx"),
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
            | FutDailyRx
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
        )

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

    @classmethod
    def new(cls, engine: EngineRoot):
        root = FutDailyRxDomain.Root.new(engine)
        return FutDailyRxDomain(root)
    
#  ------------------

# TODO: if __name__ == __main__:
# run domain by name