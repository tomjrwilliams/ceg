from __future__ import annotations
import os
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
import datetime
import datetime as dt

import functools
from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import polars as pl

from .... import apis
from ....apis import t, kv, assert_is
from ....engine import *

from ..bars.d import *
from ..calc.d import *
from ..eco.d import *

from ..agg import *
from ..pfo import *

#  ------------------


@dataclass(frozen=True)
class PxWeight:
    clip: tuple[float, float]

    def label(self) -> str:
        return ""

    def apply(
        self,
        px: Daily,
        rx: bool = False,
        # if rx, then return rx
    ):
        f = "px" if not rx else "rx"
        # TODO
        return DailyTF.new(
            px,
            f=f,
        )


@dataclass(frozen=True)
class RxWeight:
    def label(self) -> str:
        return ""

    def apply(
        self,
        rx: Daily,
    ):
        f = "rx"
        # TODO
        return DailyTF.new(
            rx,
            f=f,
        )


@dataclass(frozen=True)
class AssetConcrete:
    product: str
    symbol: str
    pct: bool = False

    def unpack(self, *args, **kwargs):
        return self

    def price_symbols(self):
        return [self.symbol]

    def price_types(self):
        return [type(self.rx())]

    def get_key(self):
        return self.label()

    def label(self):
        pct = "-pct" if self.pct else ""
        return f"{self.symbol}{pct}"

    def gross(self):
        return self.px()

    def px(
        self,
    ):

        if self.product == "FUT":
            scale = FutDailyOrdinal.new(
                symbol=self.symbol,
                ident=1,
                field="close",
                adjust="raw",
                source="frd",
            )
            return scale
        elif self.product == "ETF":
            scale = EqDailyEtf.new(
                symbol=self.symbol,
                field="close",
                adjust="price",
                source="frd",
            )
            return scale
        elif self.product == "FX":
            scale = FxDailyPx.new(
                symbol=self.symbol,
                field="close",
                # adjust="price",
                source="frd",
            )
            return scale

    def rx(
        self,
    ):
        return assert_is(
            FutDailyRx.new(
                symbol=self.symbol,
                ident=1,
                field="close",
                adjust="abs",
                source="frd",
            )
            if self.product == "FUT"
            else EqDailyRx.new(
                symbol=self.symbol,
                field="close",
                adjust="full",
                source="frd",
            )
            if self.product == "ETF"
            else FxDailyRx.new(
                symbol=self.symbol,
                field="close",
                # adjust="full",
                source="frd",
            )
            if self.product == "FX"
            else None
        )

    # def with_rx(
    #     self,
    #     params: BtParams,
    #     result: BtResult,
    #     universe: ContainerDaily,
    # ):
    #     rx = self.rx(eng)
    #     return universe.with_rx(
    #         rx, into=result
    #     )


@dataclass(frozen=True)
class AssetSynthetic:
    id: str
    alias: str | None = None
    symbol: str | None = None  # pfo key

    def price_types(self):
        rx = self.rx()
        if isinstance(
            rx,
            (
                FutDailyRx,
                EqDailyRx,
                FxDailyRx,
            ),
        ):
            return [type(rx)]
        elif isinstance(rx, DailyBtRx):
            return rx.get_parent().price_types()
        elif isinstance(rx, DailyPfolioRx):
            raise ValueError(rx)
        else:
            raise ValueError(rx)

    def price_symbols(self):
        if self.symbol is not None:
            return [self.symbol]
        rx = self.rx()
        if isinstance(
            rx,
            (
                FutDailyRx,
                EqDailyRx,
                FxDailyRx,
            ),
        ):
            px = rx.get_parent()
            return [px.tags.symbol]
        elif isinstance(rx, DailyBtRx):
            return rx.get_parent().price_symbols()
        elif isinstance(rx, DailyPfolioRx):
            raise ValueError(rx)
        else:
            raise ValueError(rx)

    # @property
    # def symbol(self):
    #     return self.key or self.label()

    def get_key(self):
        return None

    def label(self):
        return self.alias or self.id

    def rx(self):
        return context.get().get(self.id, t=Daily)

    def gross(self):
        rx = self.rx()
        if isinstance(rx, DailyBtRx):
            return rx.get_parent().gross()
        elif isinstance(
            rx,
            (
                FutDailyRx,
                EqDailyRx,
                FxDailyRx,
            ),
        ):
            px = rx.get_parent()
            return px
        else:
            raise ValueError(self)

    def px(self):
        rx = self.rx()
        if isinstance(rx, DailyBtRx):
            return rx.get_parent().px()
        price_symbols = self.price_symbols()
        price_types = self.price_types()
        assert len(price_symbols) == 1, price_symbols
        assert len(price_types) == 1, price_types
        symbol = price_symbols[0]
        t = price_types[0]
        if t in (
            FutDailyOrdinal,
            FutDailyRx,
        ):
            scale = FutDailyOrdinal.new(
                symbol=symbol,
                ident=1,
                field="close",
                adjust="raw",
                source="frd",
            )
            return scale
        elif t in (EqDailyEtf, EqDailyRx):
            scale = EqDailyEtf.new(
                symbol=symbol,
                field="close",
                adjust="price",
                source="frd",
            )
            return scale
        elif t in (FxDailyPx, FxDailyRx):
            scale = FxDailyPx.new(
                symbol=symbol,
                field="close",
                # adjust="price",
                source="frd",
            )
            return scale
        else:
            raise ValueError(
                dict(
                    self=self,
                    price_type=t,
                    symbol=symbol,
                )
            )

    # def with_rx(self,
    #     params: BtParams,
    #     result: BtResult,
    #     universe: ContainerDaily,
    # ):
    #     assert result.domain is not None, result
    #     rx = context.get().get(self.name, t=Daily)
    #     assert rx is not None, self
    #     return universe.with_rx(
    #         rx, into=result
    #     )


@dataclass(frozen=True)
class AssetLike:
    pass


@dataclass(frozen=True)
class HasAsset:
    asset: (
        AssetConcrete
        | AssetSynthetic
        | tuple[AssetLike, ...]
    )


__all__ = [
    "PxWeight",
    "RxWeight",
    "AssetLike",
    "AssetConcrete",
    "AssetSynthetic",
    "HasAsset",
]
