
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
from .....apis import t
from .....apis import kv
from .....engine import *

from .root import *

#  ------------------

@dataclass(frozen=True)
class EcoDailyObservation(
    HasEcoDailyParent,
    DailyChunked,
):
    parent: EcoDaily | str

    tags: EcoDailyTags

    version: int = 0
    type: Literal["EcoDailyObservation"] = "EcoDailyObservation"

    @dataclass(frozen=True)
    class Snap(EcoDaily.Snap):
        type: Literal["EcoDailyObservationSnap"] = "EcoDailyObservationSnap"
 
    def get_snap(
        self, domain: Domain
    ) -> EcoDailyObservation.Snap | None:
        return self.snap_key_val(domain).get(self.path, EcoDailyObservation.Snap)
    
    @classmethod
    def new(
        cls,
        parent: EcoDaily | EngineRoot,
        symbol: str,
        units: str | None = None,
        frequency: str | None = None,
        seasonal: str | None = None,
        source: str | None = None,
        asof: str | None = None,
        lag: dt.timedelta | None = None,
    ):
        s_lag: str | None = None

        if ":" in symbol:
            symbol, *asof_lag = symbol.split(":")
            for s in asof_lag:
                if lag is None and s[0] == "L":
                    s_lag = s
                    lag = apis.cal.parse_lag(s_lag)
                elif asof is None and s[0] != "L":
                    asof = s
            
        if isinstance(parent, EngineRoot):
            parent = EcoDaily.new(parent)
        if units is None:
            if source == "fred":
                s = apis.data.fred.load_spec(
                    symbol,
                    os.environ.get(
                        "FRED_DATA", "./__local__/fred"
                    )
                )
                units = s.units
                frequency = s.frequency
                seasonal = s.seasonal_adjustment
            else:
                raise ValueError((symbol, source))
        assert frequency is not None, symbol
        assert seasonal is not None, seasonal
        assert source is not None, source
        ident = f"{symbol}-{source}"
        if asof is not None:
            ident = f"{ident}-{asof}"
        if s_lag is not None:
            ident = f"{ident}-{s_lag}"
        return cls(
            path=parent.extend_path(ident),
            parent=parent,
            tags=EcoDailyTags(
                symbol=symbol,
                source=source,
                units=units,
                frequency=frequency,
                seasonal=seasonal,
                asof=asof,
                lag=lag,
            ),
        )

    # resampling etc.?

    def depends(self, domain: Domain) -> list[Dependency]:
        return (
            self.depends_self(domain)
            + self.depends_domain(domain)
        )

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        res: list[Handler] = []
        if isinstance(event, DomainSync):
            res.append(HandleUpdateEcoDailyObservation())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewEcoDailyObservation())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateEcoDailyObservation(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: EcoDailyObservation,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewEcoDailyObservation(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: EcoDailyObservation,
        domain: Domain,
    ):
        tags = obj.tags
        snap = obj.get_snap(domain)

        if snap is not None:
            return None
            # # NOTE: should already be handled?
            # raise ValueError((self, event))

        if tags.source == "fred":
            df = apis.data.fred.load_time_series(
                tags.symbol,
                os.environ.get(
                    "FRED_DATA", "./__local__/fred"
                ),
                dense=True,
                asof=tags.asof,
                lag=tags.lag,
            ).with_columns(
                pl.col(tags.symbol)
                # .forward_fill()
            )
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
            tags.symbol,
            chunk=kv.BODY,
            publish=True,
        )

        snap = (
            obj.snap_key_val(domain)
            .set(obj.path, EcoDailyObservation.Snap(
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
class EcoDailyDomainOnlyObservation(EcoDailyDomainBase):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | EcoDaily
            | EcoDailyObservation
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: EcoDailyObservation.Snap

@dataclass(frozen=True)
class EcoDailyObservationDomain(EcoDailyDomainOnlyObservation):
    """
    >>> eng = EngineRoot.new(ROOT)
    >>> obj = EcoDailyObservation.new(
    ...     eng,
    ...     symbol="VIXCLS",
    ...     source="fred",
    ... )
    >>> domain = (
    ...    EcoDailyObservationDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    shape: (8,)
    Series: 'test-eco-d-VIXCLS-fred' [f64]
    [
        17.21
        16.35
        null
        null
        17.24
        16.59
        16.08
        15.78
    ]
    >>> obj = EcoDailyObservation.new(
    ...     eng,
    ...     symbol="US_CAP_UTIL",
    ...     source="fred",
    ...     asof="3W",
    ... )
    >>> domain = (
    ...    EcoDailyObservationDomain.empty(eng)
    ...     .with_obj(obj)
    ...     .sync()
    ... )
    >>> obj.value_key_val(domain).get(obj.path, 1).tail(8)
    """
    @dataclass(frozen=True)
    class Root(EcoDailyDomainBase.Root):
        parent: EcoDaily | str
        version: int = 0
        type: Literal["EcoDailyObservationDomain"] = "EcoDailyObservationDomain"

        SUFFIX: ClassVar[str] = "eco"

    @classmethod
    def empty(cls, engine: EngineRoot):
        root = EcoDailyObservationDomain.Root.new(engine)
        return EcoDailyObservationDomain(root)

    @classmethod
    def new(
        cls, symbol: str = "VIXCLS"
    ) -> EcoDailyObservationDomain:
        eng = EngineRoot.new(ROOT)
        self = EcoDailyObservationDomain.empty(eng)
        obj = EcoDailyObservation.new(
            eng,
            symbol=symbol, 
            source="fred"
        )
        self = self.with_obj(obj)
        return cast(EcoDailyObservationDomain, self)
    
@dataclass(frozen=True)
class HasEcoDailyObservationParent(Object):
    parent: EcoDailyObservation | str

    def get_parent(self, domain: Domain) -> EcoDailyObservation:
        return domain.get(self.parent, EcoDailyObservation)

#  ------------------

__all__ = [
    "EcoDailyObservation",
    "HandleUpdateEcoDailyObservation",
    "HandleNewEcoDailyObservation",
    "EcoDailyDomainOnlyObservation",
    "EcoDailyObservationDomain",
    "HasEcoDailyObservationParent",
]
# TODO: if __name__ == __main__:
# run domain by name