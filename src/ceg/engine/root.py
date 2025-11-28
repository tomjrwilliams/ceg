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

# TODO: os.environ.get()
ROOT = "test"
ROOT_DB = "./__local__/test.db"

SQLite = "SQLite"

# TODO: rename environment??

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

    def get_config(
        self, k: str, default: str = "events"
    ) -> EngineStorageConfig:
        config = getattr(self, k)
        if isinstance(config, EngineStorageConfig):
            return config
        config = getattr(self, default)
        assert isinstance(config, EngineStorageConfig), (
            config, k, default
        )
        return config

@dataclass(frozen=True)
class ParentHasRoot(Object):

    def root(self, domain: Domain) -> EngineRoot: ...

@dataclass(frozen=True)
class UseRootConfigs(Object):

    def get_parent(
        self, domain: Domain
    ) -> ParentHasRoot:
        assert self.parent is not None, self
        return cast(
            ParentHasRoot,
            domain.get(self.parent, t=Object)
        )

    def root(self, domain: Domain) -> EngineRoot:
        if isinstance(self, EngineRoot):
            return self
        return self.get_parent(domain).root(domain)

    def event_queue(self, domain: Domain) -> EventQueue: 
        storage = self.root(domain).storage
        db = storage.get_config("events")
        table = db.table or "Events"
        if db.type == SQLite:
            return EventQueueSQLite.new(
                db=db.conn,
                table=table,
                parser=domain.event_parser
            )
        raise ValueError(self, db)

    def dependency_key_val(self, domain: Domain) -> DependencyKeyVal:
        storage = self.root(domain).storage
        db = storage.get_config("depends")
        table = db.table or "Depends"
        if db.type == SQLite:
            return DependencyKeyValSQLite.new(
                db=db.conn,
                table=table,
                parser=domain.depends_parser
            )
        raise ValueError(self, db)

    def obj_key_val(self, domain: Domain) -> ObjectKeyVal:
        storage = self.root(domain).storage
        db = storage.get_config("objects")
        table = db.table or "Objects"
        if db.type == SQLite:
            return ObjectKeyValSQLite.new(
                db=db.conn,
                table=table,
                parser=domain.object_parser
            )
        raise ValueError(self, db)

    def snap_key_val(self, domain: Domain) -> SnapKeyVal:
        storage = self.root(domain).storage
        db = storage.get_config("snaps")
        table = db.table or "Snaps"
        if db.type == SQLite:
            return SnapKeyValSQLite.new(
                db=db.conn,
                table=table,
                parser=domain.snap_parser
            )
        raise ValueError(self, db)

    def value_key_val(self, domain: Domain) -> ValueKeyVal:
        storage = self.root(domain).storage
        # NOTE: override with eg. numpy store as needed
        db = storage.get_config("values")
        table = db.table or "Values"
        if db.type == SQLite:
            return ValueKeyValJSONSQLite.new(
                db=db.conn,
                table=table,
                parser=domain.value_parser
            )
        raise ValueError(self, db)

@final
@dataclass(frozen=True)
class EngineRoot(UseRootConfigs):
    parent: Object | None = Object.parent_field()
    storage: EngineStorage = Field()
    version: int = 0
    type: Literal["EngineRoot"] = "EngineRoot"

    def root(self, domain: Domain):
        return self
    
    @classmethod
    def new(
        cls,
        root: str | None = None,
        storage: EngineStorage | str | None = None,
    ) -> EngineRoot:
        root = root or str(ROOT)
        assert root is not None, root
        storage = storage or str(ROOT_DB)
        if isinstance(storage, str):
            storage = EngineStorage(
                events=EngineStorageConfig(
                    storage,
                    type=SQLite
                )
            )
        return EngineRoot(
            path=root,
            storage=storage,
            parent=None,
        )

    def depends(self, domain: Domain) -> list[Dependency]:
        return []

    def handlers(self, event: Event, domain: Domain) -> list[Handler]:
        return []

@dataclass(frozen=True)
class UseChunkedValues(UseRootConfigs):

    @property
    def chunk(self) -> int: ...

    def value_key_val(
        self,
        domain: Domain,
        t: None = None,
        # TODO: return union with parquet etc.
        # can then create a parser discriminator class
    ) -> (
        ValueKeyChunkNumPySQLite
        | ValueKeyChunkPlSeriesSQLite
    ):
        storage = self.root(domain).storage
        db = storage.get_config("values")
        # TODO: get table name default from the key_val class itself?

        if db.type == SQLite:
            return ValueKeyChunkPlSeriesSQLite.new(
                size=self.chunk,
                db=db.conn,
            )
        # TODO: if format == numpy: ...
        # TODO: elif ...?

        raise ValueError(self, db)

@dataclass(frozen=True)
class DomainWithEngineRoot(Domain):
    
    @dataclass(frozen=True)
    class Root(UseRootConfigs):

        SUFFIX: ClassVar[str] = "TODO"

        @classmethod
        def new(
            cls, engine: EngineRoot
        ) -> DomainWithEngineRoot.Root:
            return cls(
                parent=engine,
                path=engine.extend_path(cls.SUFFIX)
            )
    
    @classmethod
    def empty(cls, engine: EngineRoot):
        root = cls.Root.new(engine)
        return cls(root)

__all__ = [
    "SQLite",
    "EngineStorageConfig",
    "EngineStorage",
    "EngineRoot",
    "UseRootConfigs",
    "UseChunkedValues",
    "ROOT",
    "ROOT_DB",
    "DomainWithEngineRoot",
]