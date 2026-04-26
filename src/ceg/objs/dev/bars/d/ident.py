from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Literal,
    Any,
    TypeVar,
    ClassVar,
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

from .....apis import kv
from .....engine import *

# TODO: dividends as themselves a type?

ORD_ADJUSTMENTS = ["pct", "abs", "raw"]

IDX_ADJUSTMENTS = ["raw"]

FX_ADJUSTMENTS = [
    # "full",
    "raw",
]

RX_ADJUSTMENTS = ["pct", "abs"]

EQ_ADJUSTMENTS = [
    "pct",
    "split",
    "raw",
    "price",  # unsplit
    "full",  # unsplit, with div adjustments
    "divs",
    "splits",
]
USD = "usd"
EUR = "eur"
JPY = "jpy"
GBP = "gbp"

CCY = set(
    [
        USD,
        EUR,
        JPY,
        GBP,
    ]
)

Ord = Literal["ord"]
Contr = Literal["contr"]
Etf = Literal["etf"]
Stock = Literal["stock"]
Fx = Literal["fx"]
Index = Literal["index"]

ORD: Ord = "ord"
CONTR: Contr = "contr"
ETF: Etf = "etf"
STOCK: Stock = "stock"
FX: Fx = "fx"
INDEX: Index = "index"

TYPES = set(
    [
        ORD,
        CONTR,
        ETF,
        STOCK,
        FX,
        INDEX,
    ]
)


def are_none(*vs: Any) -> bool:
    return all([v is None for v in vs])


def are_not_none(*vs: Any) -> bool:
    return not any([v is None for v in vs])


@dataclass(frozen=True)
class BarDailyTags:
    symbol: str
    ident: int | str
    field: str
    adjust: str
    source: str
    fx: str | None = None  # ccy to convert to


@dataclass(frozen=True)
class BarDaily(DailyChunked):
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
        expect: Etf = ETF,
    ) -> tuple[
        str, str, None, str, str, str, str | None
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
        expect: Stock = STOCK,
    ) -> tuple[
        str, str, None, str, str, str, str | None
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
        expect: Fx = FX,
    ) -> tuple[
        str, str, None, str, str, str, str | None
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
        expect: Index = INDEX,
    ) -> tuple[
        str, str, None, str, str, str, str | None
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
        str,
        str,
        int | str | None,
        str,
        str,
        str,
        str | None,
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
        expect: (
            Ord | Contr | Etf | Stock | Fx | Index | None
        ) = None,
    ) -> tuple[
        str,
        str,
        int | str | None,
        str,
        str,
        str,
        str | None,
    ]:
        if are_none(ident, field, adjust, source):
            assert expect in TYPES, expect
            assert (
                field is None
                and adjust is None
                and source is None
                and fx is None
            ), (symbol, ident, field, adjust, source, fx)
            (symbol, ident, field, adjust, source, fx) = (
                cls.parse_identifier(symbol, expect)
            )
        assert (
            field is not None
            and adjust is not None
            and source is not None
        ), (symbol, ident, field, adjust, source, fx)
        id="-".join(
            [
                symbol,
            ]
            + ([] if ident is None else [str(ident)])
            + [
                field,
                adjust,
                source,
            ]
        )
        if fx is not None:
            id += f"-{fx}"
        return (
            id,
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
        cls, symbol: str, expect: Ord
    ) -> tuple[str, int, str, str, str, str | None]: ...

    @overload
    @classmethod
    def parse_identifier(
        cls, symbol: str, expect: Contr
    ) -> tuple[str, str, str, str, str, str | None]: ...

    @overload
    @classmethod
    def parse_identifier(
        cls, symbol: str, expect: Etf
    ) -> tuple[str, None, str, str, str, str | None]: ...

    @overload
    @classmethod
    def parse_identifier(
        cls, symbol: str, expect: Stock
    ) -> tuple[str, None, str, str, str, str | None]: ...

    @overload
    @classmethod
    def parse_identifier(
        cls, symbol: str, expect: Fx
    ) -> tuple[str, None, str, str, str, str | None]: ...

    @overload
    @classmethod
    def parse_identifier(
        cls, symbol: str, expect: Index
    ) -> tuple[str, None, str, str, str, str | None]: ...

    @overload
    @classmethod
    def parse_identifier(
        cls,
        symbol: str,
        expect: None = None,
    ) -> tuple[
        str, int | str | None, str, str, str, str | None
    ]: ...

    @classmethod
    def parse_identifier(
        cls,
        symbol: str,
        expect: (
            Ord | Contr | Etf | Stock | Fx | Index | None
        ) = None,
    ) -> tuple[
        str, int | str | None, str, str, str, str | None
    ]:
        original = symbol
        segs = symbol.split("-")
        fx = None
        ident = None
        if segs[-1] in CCY:
            *segs, fx = segs
        if len(segs) == 5:
            (symbol, ident, field, adjust, source) = segs
        elif len(segs) == 4:
            (symbol, field, adjust, source) = segs
        else:
            raise ValueError(original)
        if expect == ORD:
            assert isinstance(ident, str), original
            assert ident.isnumeric(), original
            ident = int(ident)
        elif expect == CONTR:
            assert isinstance(ident, str), original
            assert not ident.isnumeric(), original
        elif expect == ETF:
            assert ident is None, original
        elif expect == STOCK:
            assert ident is None, original
        elif expect == FX:
            assert ident is None, original
        elif expect == INDEX:
            assert ident is None, original
        else:
            raise ValueError(original)

        return symbol, ident, field, adjust, source, fx


TBarDaily = TypeVar("TBarDaily", bound=BarDaily)

__all__ = [
    "IDX_ADJUSTMENTS",
    "ORD_ADJUSTMENTS",
    "FX_ADJUSTMENTS",
    "EQ_ADJUSTMENTS",
    "RX_ADJUSTMENTS",
    "BarDailyTags",
    "BarDaily",
    "Ord",
    "Contr",
    "Etf",
    "Stock",
    "Fx",
    "Index",
    "ORD",
    "CONTR",
    "ETF",
    "STOCK",
    "FX",
    "INDEX",
    "TBarDaily",
]
