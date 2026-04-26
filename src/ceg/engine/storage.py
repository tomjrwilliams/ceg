from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    Callable,
    overload,
    cast,
    final,
)
from pathlib import Path
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

from ..apis import t, kv, assert_is

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *

# TODO: os.environ.get()
ROOT = "test"
ROOT_DB = "./__local__/test.db"

SQLite = "SQLite"
REDIS = "redis"

DataFrame = "dataframe"

Series = "Series"


@functools.lru_cache(maxsize=16)
def event_key_val(config: StorageConfig) -> EventKeyVal:
    if config.db == SQLite:
        return EventKeyValSQLite.new(
            db=config.conn,
            table=config.table,
            parser=context.get().event_parser,
        )
    elif config.db == REDIS:
        return EventKeyValRedis.new(
            db=config.conn,
            table=config.table,
            parser=context.get().event_parser,
        )
    raise ValueError(config)


@functools.lru_cache(maxsize=16)
def dependency_key_val(
    config: StorageConfig,
) -> DependencyKeyVal:
    if config.db == SQLite:
        return DependencyKeyValSQLite.new(
            db=config.conn,
            table=config.table,
            parser=context.get().depends_parser,
        )
    elif config.db == SQLite:
        return DependencyKeyValRedis.new(
            db=config.conn,
            table=config.table,
            parser=context.get().depends_parser,
        )
    raise ValueError(config)


@functools.lru_cache(maxsize=16)
def obj_key_val(config: StorageConfig) -> ObjectKeyVal:
    if config.db == SQLite:
        return ObjectKeyValSQLite.new(
            db=config.conn,
            table=config.table,
            parser=context.get().object_parser,
        )
    elif config.db == REDIS:
        return ObjectKeyValRedis.new(
            db=config.conn,
            table=config.table,
            parser=context.get().object_parser,
        )
    raise ValueError(config)


@functools.lru_cache(maxsize=16)
def snap_key_val(config: StorageConfig) -> SnapKeyVal:
    if config.db == SQLite:
        return SnapKeyValSQLite.new(
            db=config.conn,
            table=config.table,
            parser=context.get().snap_parser,
            on_update=lambda k, snap: None,
        )
    elif config.db == REDIS:
        return SnapKeyValRedis.new(
            db=config.conn,
            table=config.table,
            parser=context.get().snap_parser,
            on_update=lambda k, snap: None,
        )
    raise ValueError(config)


TValueKeyVal = TypeVar("TValueKeyVal", bound=ValueKeyVal)


@overload
def value_key_val(
    config: StorageConfig,
    t: None = None,
) -> ValueKeyVal: ...


@overload
def value_key_val(
    config: StorageConfig,
    t: Type[TValueKeyVal],
) -> TValueKeyVal: ...


@functools.lru_cache(maxsize=32)
def value_key_val(
    config: StorageConfig,
    t: Type[TValueKeyVal] | None = None,
):
    if config.db == SQLite and config.dtype == DataFrame:
        kv = ValueKeyValDfSQLite.new(
            # size=config.chunk,
            db=config.conn,
            on_update=lambda k, snap: None,
        )
    elif config.db == REDIS and config.dtype == DataFrame:
        kv = ValueKeyValDfRedis.new(
            # size=config.chunk,
            db=config.conn,
            table=config.table,
            on_update=lambda k, snap: None,
        )
    # TODO: if format == numpy: ...
    elif (
        config.db == SQLite and config.chunk is not None
        # and config.dtype == Series
    ):
        kv = ValueKeyChunkPlSeriesSQLite.new(
            size=config.chunk,
            db=config.conn,
            on_update=lambda k, snap, chunk: None,
        )
    elif (
        config.db == REDIS and config.chunk is not None
        # and config.dtype == Series
    ):
        kv = ValueKeyChunkPlSeriesRedis.new(
            size=config.chunk,
            db=config.conn,
            table=config.table,
            on_update=lambda k, snap, chunk: None,
        )
    elif config.db == SQLite:
        kv = ValueKeyValJSONSQLite.new(
            db=config.conn,
            table=config.table,
            parser=context.get().value_parser,
            on_update=lambda k, snap: None,
        )
    elif config.db == REDIS:
        kv = ValueKeyValJSONRedis.new(
            db=config.conn,
            table=config.table,
            parser=context.get().value_parser,
            on_update=lambda k, snap: None,
        )
    else:
        raise ValueError(config)
    if t is None:
        return assert_is(kv, ValueKeyVal)
    return assert_is(kv, t)


@dataclass(frozen=True)
class UseRootStorage(Object):
    # TODO: some kind of caching at grpah level?

    def event_key_val(self, mode: str = "r") -> EventKeyVal:
        root = self.id.split("-", maxsplit=1)[0]
        return event_key_val(
            context.get().get(
                f"config-storage-{root}-event-{mode}",
                StorageConfig,
            )
        )

    def dependency_key_val(self, mode: str = "r") -> DependencyKeyVal:
        root = self.id.split("-", maxsplit=1)[0]
        return dependency_key_val(
            context.get().get(
                f"config-storage-{root}-dependency-{mode}",
                StorageConfig,
            )
        )

    def obj_key_val(self, mode: str = "r") -> ObjectKeyVal:
        root = self.id.split("-", maxsplit=1)[0]
        try:
            return obj_key_val(
                context.get().get(
                    f"config-storage-{root}-object-{mode}",
                    StorageConfig,
                )
            )
        except Exception as e:
            raise ValueError(self) from e

    def snap_key_val(self, mode: str = "r") -> SnapKeyVal:
        root = self.id.split("-", maxsplit=1)[0]
        return snap_key_val(
            context.get().get(
                f"config-storage-{root}-snap-{mode}",
                StorageConfig,
            )
        )

    @overload
    def value_key_val(
        self,
        t: None = None,
        mode: str = "r",
    ) -> ValueKeyVal: ...

    @overload
    def value_key_val(
        self,
        t: Type[TValueKeyVal],
    ) -> TValueKeyVal: ...

    def value_key_val(
        self,
        t: Type[TValueKeyVal] | None = None,
        mode: str = "r",
    ):
        root = self.id.split("-", maxsplit=1)[0]
        return value_key_val(
            context.get().get(
                f"config-storage-{root}-value-{mode}",
                StorageConfig,
            ),
            t=t,
        )


@dataclass(frozen=True)
class UseRootDataFrameStorage(UseRootStorage):
    def value_key_val(
        self,
        mode: str = "r",
        t=None,
        # TODO: t overload?
    ):
        root = self.id.split("-", maxsplit=1)[0]
        return value_key_val(
            context.get().get(
                f"config-storage-{root}-value-dataframe-{mode}",
                StorageConfig,
            ),
            # t=ValueKeyValDfSQLite,
            t=ValueKeyValDfRedis
        )


@dataclass(frozen=True)
class UseRootChunkedStorage(UseRootStorage):
    def value_key_val(
        self,
        mode: str = "r",
        t=None,
        # TODO: t overload?
    ):
        root = self.id.split("-", maxsplit=1)[0]
        return value_key_val(
            context.get().get(
                f"config-storage-{root}-value-chunked-{mode}",
                StorageConfig,
            ),
            # t=ValueKeyChunkPlSeriesSQLite,
            t=ValueKeyChunkPlSeriesRedis,
        )


# TODO: so now nothing needs listen to def changes, its handled by teh system? so we dont need to get the diff in the event

# just log that there was one

# hence events can just be name and path (so encode as just string for the path? no bytes / decode?)


@dataclass(frozen=True)
class Project(UseRootStorage):
    parent: None  # or allow?

    name: str

    # used as parent for objects you want to be able to easily wipe using the standard event apparatus
    # and parent for config objects

    version: int = 0
    type: Literal["Project"] = "Project"

    @dataclass(frozen=True)
    class Snap(Snap):
        type: Literal["ProjectSnap"] = "ProjectSnap"

    @classmethod
    def new(cls, name: str):
        return cls(parent=None, id=name, name=name)

    def set_storage_defaults(
        self,
        roots: list[str],
        read: dict[str, str] = dict(
            db=REDIS,
        ),
        write: dict[str, str] | None = None,
    ):
        if write is None:
            write = read
        DEFAULTS = [
            ("event", "Events", {}),
            ("dependency", "Deps", {}),
            ("object", "Objects", {}),
            ("snap", "Snaps", {}),
            ("value", "ValuesJSON", {}),
            ("value", "ValuesDF", dict(dtype="dataframe")),
            (
                "value",
                "ValuesSeries",
                dict(dtype="chunked", chunk=32),
            ),
        ]
        for root in roots:
            for domain, table, kwargs in DEFAULTS:
                for mode, params in dict(
                    r=read,
                    w=write,
                ).items():
                    kw = {}
                    db = params.get("db", REDIS)
                    if db == SQLite:
                        conn = params.get("conn", None)
                        if conn is None:
                            if os.environ.get("TEST") == "true":
                                env = "test"
                            else:
                                env = "db"
                            db_path = Path(f"./__local__/{env}")
                            db_path.mkdir(parents=True, exist_ok=True)
                            conn = str(db_path / f"{root}.db")
                        kw["conn"] = conn
                        kw["table"] = table
                        kw["db"] = db
                        kw["root"] = root
                        kw["mode"] = mode
                        for k, v in kwargs.items():
                            if k not in kw:
                                kw[k] = v
                        for k, v in params.items():
                            if k not in kw:
                                kw[v] = v
                    elif db == REDIS:
                        conn = params.get("conn", "localhost")
                        if os.environ.get("TEST") == "true":
                            env = "test"
                        else:
                            env = "prod"
                        kw["conn"] = conn
                        kw["table"] = f"{env}/{table}"
                        kw["db"] = db
                        kw["root"] = root
                        kw["mode"] = mode
                        for k, v in kwargs.items():
                            if k not in kw:
                                kw[k] = v
                        for k, v in params.items():
                            if k not in kw:
                                kw[v] = v
                    else:
                        raise ValueError(params)
                    context.get().with_obj(
                        StorageConfig.new(
                            self,
                            domain,
                            **kw,
                        ),
                        force=True,
                    )


@dataclass(frozen=True)
class Config(UseRootStorage):
    parent: Project | str | None

    @dataclass(frozen=True)
    class Snap(Snap):
        type: Literal["ConfigSnap"] = "ConfigSnap"


# TODO: can have separate db per project (/ event vs value etc.)
# then can manually delete, on delete obj if def changes, we check if db already gone, then skip individual delete call
# so still iterate all objects, but the delete call is a (cached?) existence check (if already deleted db)
# TODO: everything should listen to storageconfig?


@dataclass(frozen=True)
class StorageConfig(Config):
    # id={project.|None}config.{event|dep| etc.}
    domain: str  # object, event, etc.

    # TODO: field for value type (event vs dep vs obj vs val etc.?)

    # TODO: connection string, ie. with user and password? or just fp if sqlite
    conn: str
    table: str

    # TODO: replace with literal union sqlite,postgres ...
    db: str = REDIS

    chunk: int | None = None

    # also str literal uinon (dataframe, series, etc.)
    dtype: str | None = None
    mode: str | None = None

    version: int = 0
    type: Literal["StorageConfig"] = "StorageConfig"

    @dataclass(frozen=True)
    class Snap(Config.Snap):
        type: Literal["StorageConfigSnap"] = (
            "StorageConfigSnap"
        )

    def drop_table(self):
        return replace(self, table=None)

    @classmethod
    def new(
        cls,
        project: Project,
        domain: str,
        conn: str,
        table: str,
        db: str = REDIS,
        chunk: int | None = None,
        dtype: str | None = None,
        root: str | None = None,
        mode: str | None = None,
    ):
        root = root or project.name
        id=f"config-storage-{root}-{domain}"
        # TODO: put proejct, but then jhave to be able to detect if a given node is under project or global?
        # if project is not None:
        #     id=project.extend_id(path)
        if dtype is not None:
            id=f"{id}-{dtype}"
        if mode is not None:
            id=f"{id}-{mode}"
        return cls(
            id=id,
            parent=project,
            domain=domain,
            conn=conn,
            table=table,
            db=db,
            chunk=chunk,
            dtype=dtype,
            mode=mode,
        )


__all__ = [
    "Project",
    "Config",
    "ROOT",
    "ROOT_DB",
    "SQLite",
    "DataFrame",
    "Series",
    "ObjectKeyValSQLite",
    "event_key_val",
    "dependency_key_val",
    "obj_key_val",
    "snap_key_val",
    "value_key_val",
    "TValueKeyVal",
    "StorageConfig",
    "ObjectKeyValSQLite",
    "SQLite",
    "UseRootStorage",
    "UseRootChunkedStorage",
    "UseRootDataFrameStorage",
    "ROOT",
    "ROOT_DB",
]
