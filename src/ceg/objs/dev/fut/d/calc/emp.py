# probability (frequency) of hitting / exceeding exp weighted <>
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
from ...... import apis

from .em import FutDailyEM, FutDailyEMDomain, FutDailyOrdinal, EngineRoot

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class FutDailyEMP(FutDailyEM):
    pwindow: int | None = None

    @classmethod
    def params_suffix(
        cls,
        span: float | None = None,
        alpha: float | None = None,
        pwindow: int | None = None,
    ):
        i = apis.t.only_index_not_none(
            span, alpha, strict=True
        )
        if i == 0:
            assert span is not None, span
            assert span >= 1, span
            params = f"{span}"
        elif i == 1:
            assert alpha is not None, alpha
            assert 0 < alpha < 1, alpha
            params = f"{alpha}"
        else:
            raise ValueError((span, alpha))
        params = f"{cls.SUFFIX}-" + params
        if pwindow is None:
            params += "-ex"
        else:
            params += f"-{str(pwindow)}"
        return params

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
            symbol=parent.symbol,
            ident=str(ident),
            field=parent.field,
            adjust=parent.adjust,
            source=parent.source,
            fx=parent.fx,
            span=span,
            alpha=alpha,
            pwindow=pwindow,
        )

@dataclass(frozen=True)
class FutDailyEMPDomain(FutDailyEMDomain):
    pass