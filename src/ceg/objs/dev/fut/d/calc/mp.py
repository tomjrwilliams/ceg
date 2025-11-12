
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from .m import FutDailyM, FutDailyMDomain, FutDailyOrdinal, EngineRoot

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyMP(FutDailyM):
    pwindow: int | None = None

    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        pwindow: int | None = None,
    ):
        if window is None:
            params = "ex"
        else:
            params = str(window)
        if pwindow is None:
            params += "-ex"
        else:
            params += f"-{str(pwindow)}"
        return f"{cls.SUFFIX}-" + params

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
                window=window,
                pwindow=pwindow,
            )),
            parent=parent,
            symbol=parent.symbol,
            ident=str(ident),
            field=parent.field,
            adjust=parent.adjust,
            source=parent.source,
            fx=parent.fx,
            window=window,
            pwindow=pwindow,
        )

@dataclass(frozen=True)
class FutDailyMPDomain(FutDailyMDomain):
    pass