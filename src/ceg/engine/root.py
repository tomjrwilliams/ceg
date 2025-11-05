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
import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from pydantic import (
    Field,
)

from frozendict import frozendict

from ..apis import kv

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *

SQLite = "SQLite"

@dataclass(frozen=True)
class EngineStorageConfig:
    # TODO: connection string, ie. with user and password? or just fp if sqlite
    conn: str
    # TODO: replace with literal union
    type: str = SQLite
    table: str | None = None
    
    def drop_table(self):
        return replace(self, table=None)

@dataclass(frozen=True)
class EngineStorage:
    events: EngineStorageConfig
    depends: EngineStorageConfig | None = None
    objects: EngineStorageConfig | None = None
    snaps: EngineStorageConfig | None = None
    values: EngineStorageConfig | None = None

@final
@dataclass(frozen=True)
class EngineRoot(Object):
    parent: Object | None = Object.parent_field()
    storage: EngineStorage = Field()
    version: int = 0
    type: Literal["EngineRoot"] = "EngineRoot"

    @classmethod
    def new(
        cls,
        path: str,
        storage: EngineStorage,
    ) -> EngineRoot:
        return EngineRoot(
            path=path,
            storage=storage,
            parent=None,
        )

    def event_queue(self, domain: Domain) -> EventQueue: 
        db = self.storage.events
        table = db.table or "Events"
        if db.type == SQLite:
            return EventQueueSQLite.new(
                config=kv.SQLiteQueueConfig(
                    db.conn, name=table
                ),
                parser=domain.event_parser
            )
        raise ValueError(self, db)

    def dependency_key_val(self, domain: Domain) -> DependencyKeyVal:
        db = self.storage.depends or self.storage.events.drop_table()
        table = db.table or "Depends"
        if db.type == SQLite:
            return DependencyKeyValSQLite.new(
                config=kv.SQLiteKeyValConfig(
                    db.conn, name=table
                ),
                parser=domain.depends_parser
            )
        raise ValueError(self, db)

    def obj_key_val(self, domain: Domain) -> ObjectKeyVal: 
        db = self.storage.objects or self.storage.events.drop_table()
        table = db.table or "Objects"
        if db.type == SQLite:
            return ObjectKeyValSQLite.new(
                config=kv.SQLiteKeyValConfig(
                    db.conn, name=table
                ),
                parser=domain.object_parser
            )
        raise ValueError(self, db)

    def snap_key_val(self, domain: Domain) -> SnapKeyVal: 
        db = self.storage.snaps or self.storage.events.drop_table()
        table = db.table or "Snaps"
        if db.type == SQLite:
            return SnapKeyValSQLite.new(
                config=kv.SQLiteKeyValConfig(
                    db.conn, name=table
                ),
                parser=domain.snap_parser
            )
        raise ValueError(self, db)

    def value_key_val(self, domain: Domain) -> ValueKeyVal:
        # NOTE: override with eg. numpy store as needed
        db = self.storage.values or self.storage.events.drop_table()
        table = db.table or "Values"
        if db.type == SQLite:
            return ValueKeyValJSONSQLite.new(
                config=kv.SQLiteKeyValConfig(
                    db.conn, name=table
                ),
                parser=domain.value_parser
            )
        raise ValueError(self, db)

    def depends(self, domain: Domain) -> list[Dependency]:
        return []

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        return []

__all__ = [
    "SQLite",
    "EngineStorageConfig",
    "EngineStorage",
    "EngineRoot",
]