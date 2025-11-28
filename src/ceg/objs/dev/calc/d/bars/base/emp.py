# probability (frequency) of hitting / exceeding exp weighted <>
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Callable,
    Any,
    Type,
)
from ....... import apis
from .......engine import *
from .....calc import *

from ...base import *
from .....bars.d import *
from .em import *
from ..prelude import *

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class BarDailyEMP(DailyChunkedEMP):

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
        span: float | None = None,
        alpha: float | None = None,
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
                span=span,
                alpha=alpha,
                pwindow=pwindow,
            )),
            parent=parent,
            params=DailyChunkedEMPParams(
                span=span,
                alpha=alpha,
                pwindow=pwindow,
            ),
        )

__all__ = [
    "BarDailyEMP"
]