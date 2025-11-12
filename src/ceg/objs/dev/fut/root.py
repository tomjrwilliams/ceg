from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
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

Ord = Literal["ord"]
Contr = Literal["contr"]

ORD: Ord = "ord"
CONTR: Contr = "contr"

@dataclass(frozen=True)
class Fut(Object):
    parent: EngineRoot
    version: int = 0
    type: Literal["Fut"] = "Fut"

    @classmethod
    def new(cls, parent: EngineRoot, path = "fut"):
        return cls(
            path=f"{parent.path}-{path}",
            parent=parent,
        )
    
    @overload
    @classmethod
    def identifier(
        cls,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        expect: Ord = ORD,
    ) -> tuple[
        str, str, int, str, str, str, str | None
    ]: ...

    @overload
    @classmethod
    def identifier(
        cls,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        expect: Contr = CONTR,
    ) -> tuple[
        str, str, str, str, str, str, str | None
    ]: ...

    @overload
    @classmethod
    def identifier(
        cls,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        expect: None = None,
    ) -> tuple[
        str, str, int | str, str, str, str, str | None
    ]: ...

    @classmethod
    def identifier(
        cls,
        symbol: str,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        expect: Ord | Contr | None = None,
    ) -> tuple[
        str, str, int | str, str, str, str, str | None
    ]:
        if ident is None:
            assert expect == ORD or expect == CONTR, expect
            assert (
                field is None
                and adjust is None
                and source is None
                and fx is None
            ), (symbol, ident, field, adjust, source, fx)
            (
                symbol, ident, field, adjust, source, fx
            ) = Fut.parse_identifier(symbol, expect)
        assert (
            field is not None
            and adjust is not None
            and source is not None
        ), (
            symbol, ident, field, adjust, source, fx
        )
        path="-".join([
            symbol,
            str(ident),
            field,
            adjust,
            source,
        ])
        if fx is not None:
            path += f"-{fx}"
        return (
            path,
            symbol,
            ident,
            field,
            adjust,
            source,
            fx,
        )

    @overload
    @classmethod
    def parse_identifier(
        cls,
        symbol: str,
        expect: Ord
    ) -> tuple[
        str, int, str, str, str, str | None
    ]: ...
    
    @overload
    @classmethod
    def parse_identifier(
        cls,
        symbol: str,
        expect: Contr
    ) -> tuple[
        str, str, str, str, str, str | None
    ]: ...
    @overload
    @classmethod
    def parse_identifier(
        cls,
        symbol: str,
        expect: None=None,
    ) -> tuple[
        str, int | str, str, str, str, str | None
    ]: ...

    @classmethod
    def parse_identifier(
        cls,
        symbol: str,
        expect: Ord | Contr | None = None,
    ) -> tuple[
        str, int | str, str, str, str, str | None
    ]:
        segs = symbol.split(".")
        fx = None
        if len(segs) == 6:
            *segs, fx = segs
        if len(segs) == 5:
            (
                symbol,
                ord_or_con,
                field,
                adjust,
                source
            ) = segs
        else:
            raise ValueError(symbol)
        if expect == ORD:
            assert ord_or_con.isnumeric(), segs
        elif expect == CONTR:
            assert not ord_or_con.isnumeric(), segs
        elif ord_or_con.isnumeric():
            expect = ORD
        else:
            expect = CONTR
        assert expect == ORD or expect == CONTR, expect
        if expect == ORD:
            ord = int(ord_or_con)
            return symbol, ord, field, adjust, source, fx
        elif expect == CONTR:
            con = ord_or_con
            assert isinstance(con, str), con
            return symbol, con, field, adjust, source, fx
        else:
            raise ValueError(segs)
        
@dataclass(frozen=True)
class FutDaily(UseRootConfigs):
    parent: Fut | str
    version: int = 0
    type: Literal["FutDaily"] = "FutDaily"

    # TODO: freq? for when downsample to week, month etc.

    @dataclass(frozen=True)
    class Snap(Snap):
        updated: dt.datetime
        first: dt.date # most recently added
        last: dt.date

    def get_parent(self, domain: Domain) -> Fut:
        return domain.get(self.parent, Fut)

    def root(self, domain: Domain) -> EngineRoot:
        return (
            self.get_parent(domain)
            .parent
        )

    @classmethod
    def new(
        cls,
        parent: Fut | EngineRoot,
        path = "d",
    ):
        if isinstance(parent, EngineRoot):
            parent = Fut.new(parent)
        return cls(
            path=parent.extend_path(path),
            parent=parent,
        )
    
__all__ = [
    "Ord",
    "Contr",
    "ORD",
    "CONTR",
    "Fut",
    "FutDaily",
]