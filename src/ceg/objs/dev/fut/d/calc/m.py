
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

from ...... import apis
from ......apis import t
from ......apis import kv
from ......engine import *

from ....calc import *

from ...root import *
from ..bars import *


#  ------------------

@dataclass(frozen=True)
class FutDailyM(
    HasFutDailyGrandParent,
    DailyChunkedM,
):
    parent: FutDailyOrdinal | FutDailyRx | str
    # TODO: or contract
    # TODO: or rx thereof

    # tags: FutDailyTags
    params: DailyMParams

    version: int = 0
    type: Literal["FutDailyM"] = "FutDailyM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "M"
    # NOTE: to override

    @classmethod
    def new(
        cls,
        parent: FutDailyOrdinal | FutDailyRx | EngineRoot,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        # NOTE: just for contracts?
        source: str | None = None,
        fx: str | None = None,
        # TODO: expect, to decide on ordinal
        # --
        window: int | None = None,
    ):
        if isinstance(parent, EngineRoot):
            assert isinstance(symbol, str), symbol
            if ident is not None:
                assert isinstance(ident, int), ident
            parent = FutDailyOrdinal.new(
                parent,
                symbol=symbol,
                ident=ident,
                field=field,
                adjust=adjust,
                source=source,
                fx=fx,
            )
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=window,
            )),
            parent=parent,
            # tags=FutDailyTags(
            #     symbol=parent.tags.symbol,
            #     ident=parent.tags.ident,
            #     field=parent.tags.field,
            #     adjust=parent.tags.adjust,
            #     source=parent.tags.source,
            #     fx=parent.tags.fx,
            # ),
            params=DailyMParams(window=window),
        )
    
#  ------------------

__all__ = ["FutDailyM"]