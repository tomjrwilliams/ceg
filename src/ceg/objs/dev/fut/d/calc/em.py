
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
from ..bars.rx import FutDailyRx

TEST = "test"
TEST_DB = "./__local__/test.db"

#  ------------------

@dataclass(frozen=True)
class FutDailyEM(
    UseChunkedValues,
    UseRootConfigs,
):
    """FutDailyEM(UseRootConfigs):\n
    parent: FutDaily\n
    symbol: str\n
    field: str\n
    adjust: str\n
    fx: str | None = None # ccy to convert to\n
    source: str | None = None\n
    """
    parent: FutDailyOrdinal | FutDailyRx | str
    # TODO: or contract

    symbol: str
    ident: str
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to

    span: float | None = None
    alpha: float | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.

    chunk: int = 32
    # TODO: db?

    version: int = 0
    type: Literal["FutDailyEM"] = "FutDailyEM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "EM"
    # NOTE: to override

    @dataclass(frozen=True)
    class Snap(FutDaily.Snap):
        type: Literal["FutDailyEMSnap"] = "FutDailyEMSnap"
 
    def get_parent(self, domain: Domain) -> FutDailyOrdinal | FutDailyRx:
        return domain.get(self.parent, FutDailyOrdinal | FutDailyRx)

    def root(self, domain: Domain) -> EngineRoot:
        return self.get_parent(domain).root(domain)
    
    def get_snap(
        self, domain: Domain
    ) -> FutDailyEM.Snap | None:
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
        span: float | None = None,
        alpha: float | None = None,
    ):
        i = apis.t.only_index_not_none(
            span, alpha, strict=True
        )
        if i == 0:
            assert span is not None, span
            assert span >= 1, span
            params = f"{span}"
        elif i == 1:
            assert alpha is not None, alpha
            assert 0 < alpha < 1, alpha
            params = f"{alpha}"
        else:
            raise ValueError((span, alpha))
        return f"{cls.SUFFIX}-" + params

    @classmethod
    def new(
        cls,
        parent: FutDailyOrdinal | FutDailyRx | EngineRoot,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        # NOTE: just for contracts?
        source: str | None = None,
        fx: str | None = None,
        # TODO: expect, to decide on ordinal
        # --
        span: float | None = None,
        alpha: float | None = None,
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
                span=span, alpha=alpha,
            )),
            parent=parent,
            symbol=parent.symbol,
            ident=str(ident),
            field=parent.field,
            adjust=parent.adjust,
            source=parent.source,
            fx=parent.fx,
            span=span,
            alpha=alpha,
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
    ) -> HandleNewFutDailyEM:
        return HandleNewFutDailyEM()
    
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
class HandleNewFutDailyEM(Handler):

    def calc(
        self,
        event: Event,
        obj: FutDailyEM,
        domain: Domain,
        df: pl.DataFrame,
    ) -> pl.DataFrame: ...

    def handle(
        self,
        event: ObjectNew,
        obj: FutDailyEM,
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
            .set(obj.path, FutDailyEM.Snap(
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
class FutDailyEMDomain(Domain):

    @dataclass(frozen=True)
    class Root(
        UseRootConfigs,
    ):
        parent: FutDaily | str
        version: int = 0
        type: Literal["FutDailyEMDomain"] = "FutDailyEMDomain"
        # TODO: to override

        SUFFIX: ClassVar[str] = "EM"
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
            | FutDailyRx
            # | FutDailyEM
            # TODO: to override
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: (
            FutDailyOrdinal.Snap
            | FutDailyRx.Snap
            # | FutDailyEM.Snap
            # TODO: to override
        )

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        value: ValueNumPy | ValuePlSeries

    @classmethod
    def new(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)
    
#  ------------------

# TODO: if __name__ == __main__:
# run domain by name

# ewma, median, midrange, midhinge (on quartiles)