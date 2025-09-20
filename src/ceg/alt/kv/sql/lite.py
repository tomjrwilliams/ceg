
from typing import Type, Generic, Literal, TypeGuard
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod

from ..core.store import KeyValStore, KeyValStoreConfig
from ....apis import sqlite

@dataclass(frozen=True)
class SQLiteConfig(KeyValStoreConfig):
    
    db: str
    name: str
    # if plural, dict, then overload on string literals when subclass (?)
    table: sqlite.Table | None = None
    spec: dict | None = None

class TSQLite:
    Key = int | str
    Val = bool | int | float | str | bytes


@dataclass(frozen=True)
class SQLiteStore(KeyValStore[
    int,
    TSQLite.Key,
    TSQLite.Val,
]):
    
    key_type: Type[TSQLite.Key]
    val_type: Type[TSQLite.Val] | dict[str, Type[TSQLite.Val]]

    config: SQLiteConfig

    chunked: int | None = None # indicates chunk size

    def prepare(self):
        # TODO STRICT
        if isinstance(self.val_type, dict):
            # assert key not in val_type
            t = sqlite.Table.new(
                self.config.name,
                primary_key="id",
                key = self.key_type,
                **self.val_type,
                unique=("key",)
            )
        else:
            t = sqlite.Table.new(
                self.config.name,
                primary_key="id",
                key = self.key_type,
                val=self.val_type,
                # TODO: dict, col names
                unique=("key",)
            )

        # TODO: if exists, assert types match!
        # TODO: add chunk to spec if given, assert there if exists

        return replace(
            self, 
            config = replace(
                self.config,
                table=t,
                spec=t.create(self.config.db)
            )
        )

    def identify(self, key: TSQLite.Key) -> None | int: ...

    def lookup(self, id: int) -> TSQLite.Key: ...

    def select_val(
        self,
        key: TSQLite.Key, 
        field: str | None = None,
        chunk: int | None = None,
    ) -> None | TSQLite.Val:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
        ...

    def select_vals(
        self,
        key: TSQLite.Key, 
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> None | tuple[TSQLite.Val, ...]:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
        ...

    def iselect_val(
        self, 
        key: TSQLite.Key, 
        field: str | None = None,
        chunk: int | None = None,
    ) -> tuple[
        int | None, TSQLite.Val | None
    ]:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
        
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

        return id, val

    def iselect_vals(
        self, 
        key: TSQLite.Key, 
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> tuple[
        int | None, tuple[TSQLite.Val, ...] | None
    ]:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
        ...

    # TODO: field is either tuple or none
    # if not singular, must pass dict (even if len =1)

    def insert_val(
        self, 
        key: TSQLite.Key, 
        val: TSQLite.Val,
        field: str | None = None,
        chunk: int | None = None,
    ) -> int:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
       
        table = self.config.table
        assert table is not None, self.config

        res = table.insert(
            key=key, val=val
        )

        return res[0]["id"]

    def insert_vals(
        self, 
        key: TSQLite.Key, 
        val: tuple[TSQLite.Val, ...],
        field: tuple[str, ...] = (),
        chunk: int | None = None,
    ) -> int:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
        ...

    def update_val(
        self, 
        key: TSQLite.Key, 
        val: TSQLite.Val, 
        field: str | None = None,
        chunk: int | None = None,
        i: int | None = None
    ) -> int:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
        table = self.config.table
        assert table is not None, self.config
        if i is None:
            res = table.update(
                val=val,
                where=table.col("key") == key
            )
        else:
            res = table.update(
                val=val,
                where=table.col("id") == i
            )
        return res[0]

    def update_vals(
        self, 
        key: TSQLite.Key, 
        val: tuple[TSQLite.Val, ...],
        field: tuple[str, ...] = (),
        chunk: int | None = None,
        i: int | None = None
    ) -> int:
        """
        >>> Key = int | str
        >>> Val = bool | int | float | str | bytes
        """
        ...


i, v = SQLiteStore(int, int, SQLiteConfig("", "")).iselect_val("")