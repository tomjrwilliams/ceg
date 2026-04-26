from dataclasses import dataclass

from typing import (
    Type,
    TypeVar,
    ClassVar,
    cast,
    final,
)
from dataclasses import dataclass, replace
import numpy as np

from ...core import (
    KeyChunk,
    KeyChunkConfig,
    ExampleKeyChunk,
    # _,
    KeyO,
    ValO,
    ValT,
    #
    HEAD,
    BODY,
    TAIL,
)
from .... import sqlite
from .key_val import ID, KeyI, ValI, SQLiteKeyValConfig

# TODO: env variable
import os

TEST_DB = os.environ.get("TEST_DB", "./__local__/test.db")


@dataclass(frozen=True)
class SQLiteKeyChunkConfig(
    SQLiteKeyValConfig,
    KeyChunkConfig,
):
    pass


@dataclass(frozen=True)
class SQLiteKeyChunk(
    KeyChunk[ID, KeyI, ValI, KeyO, ValO, ValT]
):
    config: SQLiteKeyChunkConfig

    # TODO: we can in theory set this different ot generic param in the subclass def - is there a way to err that?

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    CONFIG: ClassVar[Type[SQLiteKeyChunkConfig]] = (
        SQLiteKeyChunkConfig
    )

    def prepare(self):
        # TODO: annot return sefl
        # TODO STRICT

        t = sqlite.Table.new(
            self.config.name,
            primary_key="id",
            key=self.type_key_i,
            val=self.type_val_i,
            chunk=int,
            # TODO: dict, col names
            unique=(("key", "chunk"),),
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

    def identify(
        self,
        key: KeyI,
        chunk: int,
    ) -> None | int:

        table = self.config.table
        assert table is not None, self

        res = table.select(
            [table.col("id")],
            where=[
                table.col("key") == key,
                table.col("chunk") == chunk,
            ],
        )

        if not len(res):
            return None

        assert len(res) == 1, res
        r = res[0]

        id: int = r[f"{self.config.name}.id"]
        assert isinstance(id, self.type_id), id

        return id

    def lookup(
        self, id: int
    ) -> tuple[KeyI, int] | tuple[None, None]:

        table = self.config.table
        assert table is not None, self

        res = table.select(
            [table.col("key"), table.col("chunk")],
            where=table.col("id") == id,
        )

        if not len(res):
            return None, None

        assert len(res) == 1, res
        r = res[0]

        key: KeyI = self.check_type_key_i(
            r[f"{self.config.name}.key"]
        )
        chunk: int = r[f"{self.config.name}.chunk"]

        return key, chunk

    def select(
        self,
        key: KeyI,
        chunk: int,
    ) -> None | ValI:
        i, val_i = self.iselect(key, chunk)
        return val_i

    def iselect(
        self,
        key: KeyI,
        chunk: int,
    ) -> tuple[ID | None, ValI | None]:
        table = self.config.table
        assert table is not None, self

        res = table.select(
            [table.col("id"), table.col("val")],
            where=[
                table.col("key") == key,
                table.col("chunk") == chunk,
            ],
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
        chunk: int,
    ) -> ID:

        table = self.config.table
        assert table is not None, self.config

        res = table.insert(key=key, chunk=chunk, val=val)

        id = res[0]

        assert isinstance(id, self.type_id), id
        return id

    def update(
        self,
        key: KeyI,
        val: ValI,
        chunk: int,
        i: int | None = None,
    ) -> ID:
        table = self.config.table
        assert table is not None, self.config
        if i is None:
            res = table.update(
                val=val,
                where=[
                    table.col("key") == key,
                    table.col("chunk") == chunk,
                ],
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
class ExampleSQLiteKeyChunk(
    ExampleKeyChunk,
    SQLiteKeyChunk[int, str, bytes, str, np.ndarray, float],
):
    """
    >>> j, k, l, updated = ("j", "k", "l", [])
    >>> v_arr = lambda v: np.array([v])
    >>> kv = ExampleSQLiteKeyChunk(
    ...     on_update = lambda k, v, c: updated.append((k, v))
    ... ).prepare()
    >>> kv.get(k, HEAD)
    >>> v = kv.set(k, v_arr(0.), HEAD)
    >>> assert v == v_arr(0.) == kv.get(k, HEAD)
    >>> assert (1, v_arr(0.)) == kv.iget(k, HEAD), kv.iget(k, HEAD)
    >>> _ = kv.set(j, v_arr(1.), HEAD)
    >>> _ = kv.set(k, v_arr(2.), HEAD)
    >>> assert (2, v_arr(1.)) == kv.iget(j, HEAD)
    >>> assert (1, v_arr(2.)) == kv.iget(k, HEAD)
    >>> assert updated == [(k, v_arr(2.))]
    >>> _ = kv.append_head(k, 1.)
    >>> _ = kv.append_head(k, 0.)
    >>> _ = kv.extend_head(k, np.array([2., 1.]))
    >>> _ = kv.append_tail(j, 2.)
    >>> _ = kv.append_tail(j, 3.)
    >>> _ = kv.extend_tail(j, np.array([2., 1., 0.]))
    >>> _ = kv.append_head(j, 0.)
    >>> kv.get(k, HEAD), kv.get(k, BODY)
    (array([2.]), array([1., 0., 1., 2.]))
    >>> kv.get(j, HEAD), kv.get(j, BODY), kv.get(j, TAIL)
    (array([0.]), array([1., 2., 3., 2., 1.]), array([0.]))
    """

    config: SQLiteKeyChunkConfig = SQLiteKeyChunkConfig(
        size=1, db=TEST_DB, name="ExampleKeyChunk"
    )


__all__ = [
    "SQLiteKeyChunkConfig",
    "SQLiteKeyChunk",
]
