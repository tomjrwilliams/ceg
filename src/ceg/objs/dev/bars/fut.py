from __future__ import annotations
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

from ....apis import kv
from ....engine import *

from .root import Bar

#  ------------------

@dataclass(frozen=True)
class BarFut(Object):
    parent: Bar
    version: int = 0
    type: Literal["BarFut"] = "BarFut"

    @classmethod
    def new(
        cls,
        parent: Bar,
        path = "fut",
    ):
        return cls(
            path=f"{parent.path}.{path}",
            parent=parent,
        )

@dataclass(frozen=True)
class BarFutDaily(Object):
    parent: BarFut
    version: int = 0
    type: Literal["BarFutDaily"] = "BarFutDaily"

    # TODO: freq? for when downsample to week, month etc.

    @classmethod
    def new(
        cls,
        parent: BarFut | EngineRoot,
        path = "d",
    ):
        if isinstance(parent, EngineRoot):
            bars = Bar.new(parent)
            parent = BarFut.new(bars)
        return cls(
            path=f"{parent.path}.{path}",
            parent=parent,
        )

@dataclass(frozen=True)
class BarFutIntraDay(Object):
    parent: BarFut
    version: int = 0
    type: Literal["BarFutIntraDay"] = "BarFutIntraDay"

    # TODO: freq? for which bar sampling freq

    @classmethod
    def new(
        cls,
        parent: BarFut | EngineRoot,
        path = "i",
    ):
        if isinstance(parent, EngineRoot):
            bars = Bar.new(parent)
            parent = BarFut.new(bars)
        return cls(
            path=f"{parent.path}.{path}",
            parent=parent,
        )


#  ------------------


@dataclass(frozen=True)
class BarFutDailySnap(Snap):
    updated: dt.datetime
    head: dt.date # most recently added
    tail: dt.date
    
@dataclass(frozen=True)
class BarFutDailyOrdinalSnap(Snap):
    type: Literal["BarFutDailyOrdinalSnap"] = "BarFutDailyOrdinalSnap"


@dataclass(frozen=True)
class BarFutDailyOrdinal(Object):
    parent: BarFutDaily

    symbol: str
    field: str
    adjust: str
    # expiry: str | None = None
    fx: str | None = None  # ccy to convert to
    source: str | None = None

    version: int = 0
    type: Literal["BarFutDailyOrdinal"] = "BarFutDailyOrdinal"

    @classmethod
    def new(
        cls,
        parent: BarFutDaily,
        symbol: str,
        field: str,
        adjust: str,
        # expiry: str | None = None,
        # NOTE: just for contracts?
        source: str,
        fx: str | None = None,
    ):
        path=".".join([
            parent.path,
            "ord",
            symbol,
            field,
            adjust,
            source,
        ])
        if fx is not None:
            path += f".{fx}"
        return cls(
            path=path,
            parent=parent,
            symbol=symbol,
            field=field,
            adjust=adjust,
            # expiry=expiry, # NOTE: just for contracts?
            fx=fx,
            source=source,
        )

    def value_key_val(self, domain: Domain) -> ValueKeyVal:
        # TODO: numpy time series chunked
        # TODO: numpy_chunked_ts_daily_dense
        raise ValueError(self)

    def depends(self, domain: Domain) -> list[Dependency]:
        # TODO: self, domain, ancestors (?)
        return []

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        # TODO: domain sync:
        # check snap, if new day, check end date
        # if end date < today, look for data last update
        # if > end_date, grab data and updtae

        # TODO: obj new:
        # grab data

        # TODO: obj change:
        # throw? warn?
        return []

#  ------------------

@dataclass(frozen=True)
class BarFutDailyContractSnap(Snap):
    type: Literal["BarFutDailyContractSnap"] = "BarFutDailyContractSnap"

@dataclass(frozen=True)
class BarFutDailyContract(Object):
    parent: BarFutDaily

    symbol: str
    field: str
    adjust: str | None = None
    expiry: str | None = None
    fx: str | None = None  # ccy to convert to
    source: str | None = None

    version: int = 0
    type: Literal["BarFutDailyContract"] = "BarFutDailyContract"

    @classmethod
    def new(
        cls,
        parent: BarFutDaily,
        symbol: str,
        expiry: str,
        field: str,
        adjust: str | None = None,
        fx: str | None = None,
        source: str | None = None,
        path = "con",
    ):
        return cls(
            path=f"{parent.path}.{path}",
            parent=parent,
            symbol=symbol,
            expiry=expiry,
            field=field,
            adjust=adjust,
            fx=fx,
            source=source,
        )

    def value_key_val(self, domain: Domain) -> ValueKeyVal:
        raise ValueError(self)

    def depends(self, domain: Domain) -> list[Dependency]:
        return []

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        return []

#  ------------------

@dataclass(frozen=True)
class BarFutDailyOrdinalDomain(Domain):

    @dataclass(frozen=True)
    class object_parser(ObjectParser):
        obj: (
            EngineRoot
            | Bar
            | BarFut
            | BarFutDaily
            | BarFutDailyOrdinal
        )

    @dataclass(frozen=True)
    class snap_parser(SnapParser):
        snap: BarFutDailyOrdinalSnap

    @dataclass(frozen=True)
    class value_parser(ValueParser):
        # TODO: chunked etc.
        pass

    @classmethod
    def new(cls):
        engine_root = EngineRoot.new(
            "dev", 
            storage=EngineStorage(
                events=EngineStorageConfig(
                    "./__local__/dev.db",
                    type=SQLite
                )
            )
        )
        root = BarFutDaily.new(engine_root)
        self = BarFutDailyOrdinalDomain(root)
        self = (
            self.with_obj(BarFutDailyOrdinal.new(
                root,
                symbol="ES",
                field="close",
                adjust="pct",
                source="frd"
            ))
        )
        return self

# TODO: contract

#  ------------------

# TODO: intra day ordinal, contract

#  ------------------

# TODO: if __name__ == __main__:
# run domain by name