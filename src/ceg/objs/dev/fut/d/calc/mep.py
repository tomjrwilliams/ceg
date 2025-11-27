# exp probability (ie. decay not rolling freq
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)

from ....calc import *
from ..bars import *
from .m import *

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyMEP(DailyChunkedMEP):

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
        window: int | None = None,
        palpha: float | None = None,
        pspan: float | None = None,
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
                palpha=palpha,
                pspan=pspan,
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
            params=DailyChunkedMEPParams(
                window=window,
                palpha=palpha,
                pspan=pspan,
            ),
        )

__all__ = ["FutDailyMEP"]