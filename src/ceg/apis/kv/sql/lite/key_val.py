from __future__ import annotations
from dataclasses import dataclass

from typing import (
    Type,
    TypeVar,
    ClassVar,
    final,
)
from dataclasses import dataclass, replace

from ...core import (
    KeyVal, 
    KeyValConfig, 
    ExampleKeyVal,
    # _, 
    KeyO, 
    ValO, 
    ValT
)
from .... import sqlite

# TODO: env variable
import os
TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")


@dataclass(frozen=True)
class SQLiteKeyValConfig(KeyValConfig):

    db: str
    name: str
    # if plural, dict, then overload on string literals when subclass (?)
    table: sqlite.Table | None = None
    spec: dict | None = None


ID = TypeVar("ID", bound=int)
KeyI = TypeVar("KeyI", bound=int | str)
ValI = TypeVar("ValI", bound=bool | int | float | str | bytes)


@dataclass(frozen=True)
class SQLiteKeyVal(KeyVal[ID, KeyI, ValI, KeyO, ValO]):
    """KeyVal[ID, KeyI, ValI, KeyO, ValO]"""

    config: SQLiteKeyValConfig

    # TODO: we can in theory set this different ot generic param in the subclass def - is there a way to err that?

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    chunked: int | None = None  # indicates chunk size

    CONFIG: ClassVar[Type[SQLiteKeyValConfig]] = SQLiteKeyValConfig

    @classmethod
    def new(
        cls,
        db: str,
        name: str,
        table: sqlite.Table | None = None,
        spec: dict | None = None,
    ) -> SQLiteKeyValConfig:
        return SQLiteKeyValConfig(
            db=db,
            name=name,
            table=table,
            spec=spec,
        )

    def prepare(self):
        # TODO: annot return sefl
        # TODO STRICT

        # TODO: chunk
        t = sqlite.Table.new(
            self.config.name,
            primary_key="id",
            key=self.type_key_i,
            val=self.type_val_i,
            # TODO: dict, col names
            unique=("key",),
        ).create(self.config.db)

        # TODO: composite pkey?
        # if the type_key_i is a tuple?

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

    def identify(
        self,
        key: KeyI,
    ) -> None | int:
    
        table = self.config.table
        assert table is not None, self

        res = table.select(
            [table.col("id")],
            where=table.col("key") == key,
        )

        if not len(res):
            return None

        assert len(res) == 1, res
        r = res[0]

        id: int = r[f"{self.config.name}.id"]
        assert isinstance(id, self.type_id), id

        return id

    def lookup(self, id: int) -> KeyI:
        
        table = self.config.table
        assert table is not None, self

        res = table.select(
            [table.col("key")],
            where=table.col("id") == id,
        )

        # if not len(res):
        #     return None

        assert len(res) == 1
        key_i = res[0][f"{self.config.name}.key"]
        
        assert isinstance(key_i, self.type_key_i), key_i
        return key_i

    def select(
        self,
        key: KeyI,
    ) -> None | ValI:
        table = self.config.table
        assert table is not None, self

        res = table.select(
            [table.col("val")],
            where=table.col("key") == key,
        )

        if not len(res):
            return None

        assert len(res) == 1, res
        r = res[0]

        val = r[f"{self.config.name}.val"]
        assert isinstance(val, self.type_val_i), val

        return val

    def iselect(
        self,
        key: KeyI,
    ) -> tuple[ID | None, ValI | None]:
        table = self.config.table
        assert table is not None, self

        res = table.select(
            [table.col("id"), table.col("val")],
            where=table.col("key") == key,
        )

        if not len(res):
            return None, None

        assert len(res) == 1, res
        r = res[0]

        id: int = r[f"{self.config.name}.id"]
        val = r[f"{self.config.name}.val"]

        assert isinstance(id, self.type_id), id
        assert isinstance(val, self.type_val_i), val

        return id, val

    def insert(
        self,
        key: KeyI,
        val: ValI,
    ) -> ID:

        table = self.config.table
        assert table is not None, self.config

        res = table.insert(key=key, val=val)

        id = res[0]

        assert isinstance(id, self.type_id), id
        return id

    def update(
        self,
        key: KeyI,
        val: ValI,
        i: int | None = None,
    ) -> ID:
        table = self.config.table
        assert table is not None, self.config
        if i is None:
            res = table.update(
                val=val, where=table.col("key") == key
            )
        else:
            res = table.update(
                val=val, where=table.col("id") == i
            )

        id = res[0]
        assert isinstance(id, self.type_id), id

        return id

@final
@dataclass(frozen=True)
class ExampleSQLiteKeyVal(
    ExampleKeyVal,
    SQLiteKeyVal[int, str, int, str, float]
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> kv = ExampleSQLiteKeyVal(
    ...     on_update = lambda k, v: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k)
    >>> v = kv.set(k, 0.0)
    >>> assert v == 0.0 == kv.get(k), kv.get(k)
    >>> assert (1, 0.0) == kv.iget(k), kv.iget(k)
    >>> v = kv.set(j, 1.0)
    >>> v = kv.set(k, 2.0)
    >>> assert (2, 1.0) == kv.iget(j)
    >>> assert (1, 2.0) == kv.iget(k)
    >>> assert updated == [(k, 2.0)]
    """
    config: SQLiteKeyValConfig = SQLiteKeyValConfig(
        db=TEST_DB, name="ExampleKeyVal"
    )


__all__ = [
    "SQLiteKeyValConfig",
    "ID",
    "KeyI",
    "ValI",
    "SQLiteKeyVal",
]
