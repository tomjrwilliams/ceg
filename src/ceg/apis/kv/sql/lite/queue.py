from __future__ import annotations
from typing import (
    Type,
    TypeVar,
    cast,
    final,
)
import datetime
import datetime as dt
from dataclasses import dataclass, replace

from .... import sqlite
from ....t import (
    DtI,
    DtO,
    ValO,
)
from ...core import (
    Queue,
    QueueConfig,
    Comp,
    ExampleQueue,
)


# TODO: env variable
import os
TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")



@dataclass(frozen=True)
class SQLiteQueueConfig(QueueConfig):
    # TODO: abstract out with the other sqlite configs

    db: str
    name: str
    # if plural, dict, then overload on string literals when subclass (?)
    table: sqlite.Table | None = None
    spec: dict | None = None


ID = TypeVar("ID", bound=int)
ValI = TypeVar("ValI", bound=bool | int | float | str | bytes)


@dataclass(frozen=True)
class SQLiteQueue(Queue[ID, DtI, ValI, DtO, ValO]):
    """Queue[ID, DtI, ValI, DtO, ValO]"""
    config: SQLiteQueueConfig
    type_id: Type[ID]
    type_dt_i: Type[DtI]
    type_val_i: Type[ValI]
    type_dt_o: Type[DtO]
    type_val_o: Type[ValO]

    # TODO: other fields?
    # so we can filter on query in where, rather than post load?

    def prepare(self) -> SQLiteQueue:
        # TODO astract out with the other sqlite

        # TODO: chunk
        t = sqlite.Table.new(
            self.config.name,
            primary_key="id",
            val=self.type_val_i,
            dt=self.type_dt_i
        ).create(self.config.db)

        # TODO: if exists, assert types match!
        # TODO: add chunk to spec if given, assert there if exists

        return replace(
            self,
            config=replace(
                self.config,
                table=t,
                spec=t.pragma,
            ),
        )

    def slice(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtI]] | None = None,
        head: bool = True,
    ) -> list[ValI]:
        i_v_t = self.islice(
            n=n,
            where_i=where_i,
            where_dt=where_dt,
            head=head,
        )
        if len(i_v_t) == 0:
            return []
        i, v, t = zip(*i_v_t)
        return cast(list[ValI], list(v))

    def islice(
        self, 
        n: int = 1,
        where_i: list[tuple[Comp, ID]] | None = None,
        where_dt: list[tuple[Comp, DtI]] | None = None,
        head: bool = True,
    ) -> list[tuple[ID, ValI, DtI]]:

        table = self.config.table
        assert table is not None, self

        where = (
            [] if where_i is None else [
                getattr(sqlite.where, comp)(
                    table.col("id"), i
                )
                for comp, i in where_i
            ]
        ) + (
            [] if where_dt is None else [
                getattr(sqlite.where, comp)(
                    table.col("dt"), dt
                )
                for comp, dt in where_dt
            ]
        )

        res = table.select(
            [
                table.col("id"),
                table.col("val"),
                table.col("dt")
            ],
            where=where,
            order=getattr(
                sqlite.order, "asc" if head else "desc"
            )(table.col("id"))
        )

        if not len(res):
            return []

        if n > 0:
            assert len(res) <= n, res

        return [
            (
                self.check_type_id(
                    r[f"{self.config.name}.id"]
                ),
                self.check_type_val_i(
                    r[f"{self.config.name}.val"]
                ),
                self.check_type_dt_i(
                    r[f"{self.config.name}.dt"]
                ),
            )
            for r in res
        ]

    def insert(
        self,
        val: ValI,
    ) -> tuple[ID, DtI]:

        dt = self.encode_dt(self.init_dt())

        table = self.config.table
        assert table is not None, self.config

        res = table.insert(dt=dt, val=val)

        id = res[0]

        assert isinstance(id, self.type_id), id
        return id, dt


@final
@dataclass(frozen=True)
class ExampleSQLiteQueue(
    ExampleQueue,
    SQLiteQueue[int, float, int, dt.datetime, float]
):
    """
    >>> import time
    >>> j, k = ("j", "k")
    >>> kv = ExampleSQLiteQueue().prepare()
    >>> kv.head(1)
    []
    >>> kv.tail(1)
    []
    >>> v = kv.append(0.)
    >>> assert v == 0.0 == kv.head(1)[0]
    >>> assert ([1], [0.0]) == kv.ihead(1)[:2]
    >>> time.sleep(0.1)
    >>> dt = datetime.datetime.now()
    >>> time.sleep(0.1)
    >>> _ = kv.append(1.)
    >>> _ = kv.append(2.)
    >>> kv.tail(n=3, where_dt=[("gt", dt)])
    [2.0, 1.0]
    >>> kv.head(n=3, where_dt=[("gt", dt)])
    [1.0, 2.0]
    >>> kv.tail(n=3, where_dt=[("lt", dt)])
    [0.0]
    >>> kv.head(n=3, where_dt=[("lt", dt)])
    [0.0]
    >>> kv.tail(n=3, where_i=[("ge", 2)])
    [2.0, 1.0]
    >>> kv.head(n=3, where_i=[("ge", 2)])
    [1.0, 2.0]
    >>> kv.tail(n=3, where_i=[("le", 2)])
    [1.0, 0.0]
    >>> kv.head(n=3, where_i=[("le", 2)])
    [0.0, 1.0]
    """
    config: SQLiteQueueConfig = SQLiteQueueConfig(
        db=TEST_DB, name="ExampleQueue"
    )

__all__ = [
    "SQLiteQueue",
    "SQLiteQueueConfig",
]

# vs = Ex(SQLiteConfig("", "")).head(1)
# ids, dts, vs = Ex(SQLiteConfig("", "")).ihead(1)