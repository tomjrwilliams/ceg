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

# Ord = Literal["ord"]
# Contr = Literal["contr"]
# Etf = Literal["etf"]
# Stock = Literal["stock"]
# Fx = Literal["fx"]
# Index = Literal["index"]

# ORD: Ord = "ord"
# CONTR: Contr = "contr"
# ETF: Etf = "etf"
# STOCK: Stock = "stock"
# FX: Fx = "fx"
# INDEX: Index = "index"

# TYPES = set(
#     [
#         ORD,
#         CONTR,
#         ETF,
#         STOCK,
#         FX,
#         INDEX,
#     ]
# )


def are_none(*vs: Any) -> bool:
    return all([v is None for v in vs])


def are_not_none(*vs: Any) -> bool:
    return not any([v is None for v in vs])

__all__ = [
    "IDX_ADJUSTMENTS",
    "ORD_ADJUSTMENTS",
    "FX_ADJUSTMENTS",
    "EQ_ADJUSTMENTS",
    "RX_ADJUSTMENTS",
    # "Ord",
    # "Contr",
    # "Etf",
    # "Stock",
    # "Fx",
    # "Index",
    # "ORD",
    # "CONTR",
    # "ETF",
    # "STOCK",
    # "FX",
    # "INDEX",
]
