
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


#  ------------------

@dataclass(frozen=True)
class FutDailyM(
    UseChunkedValues,
    UseRootConfigs,
):
    """FutDailyM(UseRootConfigs):\n
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

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyM"] = "FutDailyM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "M"
    # NOTE: to override

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyMSnap"] = "FutDailyMSnap"
 
    def get_parent(self, domain: Domain) -> FutDailyOrdinal | FutDailyRx:
        return domain.get(self.parent, FutDailyOrdinal | FutDailyRx)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain).root(domain)
    
    def get_snap(
        self, domain: Domain
    ) -> FutDailyM.Snap | None:
        snap = self.snap_key_val(domain).get(self.path)
        if snap is None:
            return snap
        assert isinstance(
            snap, self.Snap
        ), snap
        return snap
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
    ):
        if window is None:
            params = "ex"
        else:
            params = str(window)
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
                window=window,
            )),
            parent=parent,
            symbol=parent.symbol,
            ident=str(ident),
            field=parent.field,
            adjust=parent.adjust,
            source=parent.source,
            fx=parent.fx,
            window=window,
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        # TODO: listen to target object
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
    
    def handler_object_new(
        self
    ) -> HandleNewFutDailyM:
        return HandleNewFutDailyM()
    
    # TODO: handler object (self) change 

    # TODO: handler update target object

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(self.handler_object_new())

        # TODO: if upstream target change

        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleNewFutDailyM(Handler):

    def calc(
        self,
        event: Event,
        obj: FutDailyM,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        
        e = (
            pl.col(obj.field).rolling_mean(
                len(df)
                if obj.window is None
                else obj.window,
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

    def handle(
        self,
        event: ObjectNew,
        obj: FutDailyM,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            return None

        parent = obj.get_parent(domain)

        parent_snap = parent.get_snap(domain)
        assert parent_snap is not None, parent_snap

        v = (
            parent.value_key_val(domain)
            .get(parent.path, kv.BODY)
        )

        # TODO: if numpy we assume dense, use snap to infer dates

        assert isinstance(v, ValuePlSeries), v
        df = self.calc(
            event,
            obj,
            domain,
            pl.DataFrame({obj.field: v.ser})
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
            .set(obj.path, FutDailyM.Snap(
                event,
                obj.path,
                dt.datetime.now(),
                parent_snap.first,
                parent_snap.last,
            ))
        )

        return e

# TODO: class handle object target change

# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------

@dataclass(frozen=True)
class FutDailyMDomain(Domain):

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: FutDaily | str
        version: int = 0
        type: Literal["FutDailyMDomain"] = "FutDailyMDomain"
        # TODO: to override

        SUFFIX: ClassVar[str] = "M"
        # TODO: to override

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
                path = parent.extend_path(cls.SUFFIX),
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
            # | FutDailyM
            # TODO: to override
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            # | FutDailyM.Snap
            # TODO: to override
        )

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)
    
#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

# ewma, median, midrange, midhinge (on quartiles)