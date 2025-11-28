
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Callable,
    Any,
    Type,
)
from .......engine import *
from ...base import *
from .....bars.d import *
from .m import *
from ..prelude import *

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class BarDailyMP(DailyChunkedMP):

    parent: (
        str
        | FutDailyOrdinal
        | FutDailyRx
        | EqDailyEtf
        | EqDailyStock
        | EqDailyRx
        | IndexDailyPx
        | IndexDailyRx
        | FxDailyPx
        | FxDailyRx
    )
    
    @classmethod
    def new(
        cls,
        parent: (
            EngineRoot
            | FutDailyOrdinal
            | FutDailyRx
            | EqDailyEtf
            | EqDailyStock
            | EqDailyRx
            | IndexDailyPx
            | IndexDailyRx
            | FxDailyPx
            | FxDailyRx
        ),
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        source: str | None = None,
        fx: str | None = None,
        # --
        window: int | None = None,
        pwindow: int | None = None,
        t: (
            None
            | Type[FutDailyOrdinal]
            | Type[FutDailyRx]
            | Type[EqDailyEtf]
            | Type[EqDailyStock]
            | Type[EqDailyRx]
            | Type[IndexDailyPx]
            | Type[IndexDailyRx]
            | Type[FxDailyPx]
            | Type[FxDailyRx]
        ) = None,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            assert isinstance(symbol, str), (cls, t)
            assert adjust is not None, adjust
            parent = t.new(
                parent,
                symbol,
                ident,
                field,
                adjust,
                source,
                fx,
            )
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=window,
                pwindow=pwindow,
            )),
            parent=parent,
            params=DailyChunkedMPParams(
                window=window,
                pwindow=pwindow,
            )
        )
    
__all__ = ["BarDailyMP"]