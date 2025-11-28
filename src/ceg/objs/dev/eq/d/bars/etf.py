
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

from ....bars import *
from ...root import *

#  ------------------


@dataclass(frozen=True)
class EqDailyEtf(
    HasEqDailyParent,
    DailyChunked,
):
    parent: EqDaily | str

    tags: EqDailyEtfTags
    
    version: int = 0
    type: Literal["EqDailyEtf"] = "EqDailyEtf"

    @dataclass(frozen=True)
    class Snap(EqDaily.Snap):
        type: Literal["EqDailyEtfSnap"] = "EqDailyEtfSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> EqDailyEtf.Snap | None:
        return self.snap_key_val(domain).get(self.path, EqDailyEtf.Snap)
    
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
            etf,
            field,
            adjust,
            source,
            fx,
        ) = Eq.identifier(
            symbol, 
            ident, 
            field, 
            adjust, 
            source, 
            fx=fx,
            expect=ETF
        )
        assert adjust in EQ_ADJUSTMENTS, adjust
        return cls(
            path=parent.extend_path(ident),
            parent=parent,
            tags=EqDailyEtfTags(
                symbol=symbol,
                ident=etf,
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
            res.append(HandleUpdateEqDailyEtf())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewEqDailyEtf())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateEqDailyEtf(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: EqDailyEtf,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewEqDailyEtf(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: EqDailyEtf,
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
                folder=apis.data.frd.Folders.ETF,
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
            .set(obj.path, EqDailyEtf.Snap(
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
class EqDailyDomainOnlyEtf(EqDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Eq
            | EqDaily
            | EqDaily
            | EqDailyEtf
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: EqDailyEtf.Snap

# SP500 trackers:
# US: SPY, VOO, and IVV
# UK/EU: VUSA, CSPX, and SPXP. 

@dataclass(frozen=True)
class EqDailyEtfDomain(EqDailyDomainOnlyEtf):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> obj = EqDailyEtf.new(
    ...     eng,
    ...     symbol="SPY",
    ...     field="close",
    ...     adjust="full",
    ...     source="frd",
    ... )
    >>> domain = (
    ...    EqDailyEtfDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-eq-d-SPY-close-full-frd' [f64]
    [
        563.98
        null
        null
        574.08
        575.46
        568.59
        567.08
        555.66
    ]
    """
    @dataclass(frozen=True)
    class Root(EqDailyDomainBase.Root):
        parent: EqDaily | str
        version: int = 0
        type: Literal["EqDailyEtfDomain"] = "EqDailyEtfDomain"

        SUFFIX: ClassVar[str] = "etf"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = EqDailyEtfDomain.Root.new(engine)
        return EqDailyEtfDomain(root)

    @classmethod
    def new(
        cls, symbol: str = "SPY"
    ) -> EqDailyEtfDomain:
        eng = EngineRoot.new(ROOT)
        self = EqDailyEtfDomain.empty(eng)
        for adj in EQ_ADJUSTMENTS:
            obj = EqDailyEtf.new(
                eng,
                symbol=symbol, 
                ident = 1,
                field="close",
                adjust=adj,
                source="frd",
            )
            self = self.with_obj(obj)
        return cast(EqDailyEtfDomain, self)
    
@dataclass(frozen=True)
class HasEqDailyEtfParent(Object):
    parent: EqDailyEtf | str

    def get_parent(self, domain: Domain) -> EqDailyEtf:
        return domain.get(self.parent, EqDailyEtf)

#  ------------------

__all__ = [
    "EqDailyEtf",
    "HandleUpdateEqDailyEtf",
    "HandleNewEqDailyEtf",
    "EqDailyDomainOnlyEtf",
    "EqDailyEtfDomain",
    "HasEqDailyEtfParent",
]
# TODO: if __name__ == __main__:
# run domain by name