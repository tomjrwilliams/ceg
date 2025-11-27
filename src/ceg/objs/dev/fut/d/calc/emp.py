# probability (frequency) of hitting / exceeding exp weighted <>
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from ...... import apis

from ....calc import *
from ..bars import *
from .em import *

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyEMP(DailyChunkedEMP):

    @classmethod
    def new(
        cls,
        parent: FutDailyOrdinal | EngineRoot,
        symbol: str | None = None,
        ident: int | str | None = None,
        field: str | None = None,
        adjust: str | None = None,
        # NOTE: just for contracts?
        source: str | None = None,
        fx: str | None = None,
        # TODO: expect, to decide on ordinal
        # --
        span: float | None = None,
        alpha: float | None = None,
        pwindow: int | None = None,
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
                span=span,
                alpha=alpha,
                pwindow=pwindow,
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
            params=DailyChunkedEMPParams(
                span=span,
                alpha=alpha,
                pwindow=pwindow,
            ),
        )

__all__ = [
    "FutDailyEMP"
]