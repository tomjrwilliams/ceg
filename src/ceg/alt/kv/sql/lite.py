from typing import (
    Type,
    TypeVar,
    Generic,
    Literal,
    TypeGuard,
)
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

from ..core import KeyVal, KeyValConfig, _, KeyO, ValO
from ....apis import sqlite


@dataclass(frozen=True)
class SQLiteConfig(KeyValConfig):

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

    config: SQLiteConfig

    # TODO: we can in theory set this different ot generic param in the subclass def - is there a way to err that?

    type_id: Type[ID]
    type_key_i: Type[KeyI]
    type_val_i: Type[ValI]

    chunked: int | None = None  # indicates chunk size

    def prepare(self):
        # TODO STRICT

        # TODO: chunk
        t = sqlite.Table.new(
            self.config.name,
            primary_key="id",
            key=self.type_key_i,
            val=self.type_val_i,
            # TODO: dict, col names
            unique=("key",),
        )

        # TODO: if exists, assert types match!
        # TODO: add chunk to spec if given, assert there if exists

        return replace(
            self,
            config=replace(
                self.config,
                table=t,
                spec=t.create(self.config.db),
            ),
        )

    def identify(
        self,
        key: KeyI,
    ) -> None | int: ...

    def lookup(self, id: int) -> KeyI: ...

    def select(
        self,
        key: KeyI,
        chunk: int | None = None,
    ) -> None | ValI: ...

    def iselect(
        self,
        key: KeyI,
        chunk: int | None = None,
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
        chunk: int | None = None,
    ) -> ID:

        table = self.config.table
        assert table is not None, self.config

        res = table.insert(key=key, val=val)

        id = res[0]["id"]

        assert isinstance(id, self.type_id), id
        return id

    def update(
        self,
        key: KeyI,
        val: ValI,
        chunk: int | None = None,
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


@dataclass(frozen=True)
class Ex(SQLiteKeyVal[int, int, bool, str, str]):
    type_id: Type[int] = int
    type_key_i: Type[str] = str
    type_val_i: Type[float] = float
    type_key_o: Type[str] = str
    type_val_o: Type[str] = str

    def encode_key_type(self, t):
        return

    def encode_val_type(self, t):
        return

    def encode_key(self, k):
        return

    def encode_val(self, v):
        return

    def decode_key(self, k):
        return

    def decode_val(self, v):
        return


i, v = Ex(SQLiteConfig("", "")).iselect(0)
