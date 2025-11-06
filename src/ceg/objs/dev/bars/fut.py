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

import numpy as np
import polars as pl

from .... import apis
from ....apis import type_vars
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
    first: dt.date # most recently added
    last: dt.date
    
@dataclass(frozen=True)
class BarFutDailyOrdinalSnap(BarFutDailySnap):
    type: Literal["BarFutDailyOrdinalSnap"] = "BarFutDailyOrdinalSnap"


@dataclass(frozen=True)
class BarFutDailyOrdinal(Object):
    """BarFutDailyOrdinal(Object):
    parent: BarFutDaily
    symbol: str
    field: str
    adjust: str
    fx: str | None = None # ccy to convert to
    source: str | None = None
    """
    parent: BarFutDaily

    symbol: str
    field: str
    adjust: str
    # expiry: str | None = None
    fx: str | None = None  # ccy to convert to
    source: str | None = None

    chunk: int = 32

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
        # NOTE: chunk not in path, as doesnt change value
        # only back-end layout
        # NOTE: likewise for back-end type: np vs pq etc.
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

    def value_key_val(
        self, domain: Domain
        # TODO: return union with parquet etc.
        # can then create a parser discriminator class
    ) -> ValueKeyChunkNumPySQLite:
        
        root = self.parent.parent.parent.parent
        db = root.storage.values or root.storage.events.drop_table()

        # TODO: get table name default from the key_val class itself?

        table = db.table or "ChunksNumPyF64"
        if db.type == SQLite:
            return ValueKeyChunkNumPySQLite.new(
                size=self.chunk,
                db=db.conn,
                table=table,
            )
        raise ValueError(self, db)

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
            res.append(HandleUpdateBarFutDailyOrdinal())
        elif isinstance(event, ObjectNew):
            assert self.path == event.obj, (self, event)
            res.append(HandleNewBarFutDailyOrdinal())
        elif isinstance(event, ObjectChange):
            # TODO: throw? unless only chunk change
            # then just rechunk
            # or backend type / loc, then move
            raise ValueError((self, event))
        return res

@dataclass(frozen=True)
class HandleUpdateBarFutDailyOrdinal(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: BarFutDailyOrdinal,
        domain: Domain,
    ):
        # TODO:
        # check latest update, if today, skip
        # check new data available
        # if after snap.last, get and append
        return

@dataclass(frozen=True)
class HandleNewBarFutDailyOrdinal(Handler):
    def handle(
        self,
        event: ObjectNew,
        obj: BarFutDailyOrdinal,
        domain: Domain,
    ):
        snap = obj.get_snap(domain)

        if snap is not None:
            # NOTE: should already be handled?
            raise ValueError((self, event))

        if obj.source == "frd":
            df = apis.data.frd.read_file(
                "TODO",
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

        d_min = df.select(pl.col("date").min()).item()
        d_max = df.select(pl.col("date").min()).item()
        assert isinstance(d_min, dt.date), d_min
        assert isinstance(d_max, dt.date), d_max

        kv_val = obj.value_key_val(domain)
        kv_snap = obj.snap_key_val(domain)

        # TODO: assert has length
        
        if isinstance(kv_val, ValueKeyChunkNumPySQLite):
            # : F64_1D
            v_np: np.ndarray = df.select(obj.field).to_numpy().T[0]
            v = ValueNumPy(obj.path, type_vars.ndarray.cast(v_np))
            v = kv_val.set(obj.path, v, kv.BODY)
            
        elif isinstance(kv_val, ValueKeyChunkPlSeriesParq):
            v_s = df.select(obj.field).to_series(0)
            v = ValuePlSeries(obj.path, type_vars.pl.Series.cast(v_s))
            v = kv_val.set(obj.path, v, kv.BODY)

        else:
            raise ValueError(self, event)

        e = obj.publish(ValueUpdate(obj.path), domain)

        snap = BarFutDailyOrdinalSnap(
            event,
            obj.path,
            dt.datetime.now(),
            d_min,
            d_max,
        )
        snap = kv_snap.set(obj.path, snap)

        return e

# TODO: class HandleObjectChange:
# if chunk, rechunk
# if backend type / loc change, copy
# else, throw

#  ------------------

@dataclass(frozen=True)
class BarFutDailyContractSnap(BarFutDailySnap):
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
        # TODO: as for ordinal, but specific expiry
        raise ValueError(self)

    def depends(self, domain: Domain) -> list[Dependency]:
        # TODO: as for ordinal, but specific expiry
        return []

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        # TODO: as for ordinal, but specific expiry
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
        value: ValueNumPy | ValuePlSeries

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

# TODO: contract domain

#  ------------------

# TODO: intra day
# ordinal, contract

#  ------------------

# TODO: if __name__ == __main__:
# run domain by name