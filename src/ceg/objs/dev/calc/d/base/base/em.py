
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

from ....... import apis
from .......apis import t
from .......apis import kv
from .......engine import *


#  ------------------

@dataclass(frozen=True)
class DailyEMParams(DailyParams):
    chunk: int = 32
    # TODO: db?

    span: float | None = None
    alpha: float | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.


@dataclass(frozen=True)
class DailyChunkedEM(DailyChunked):
    parent: Object | str
    # tags: DailyTags
    
    params: DailyEMParams

    version: int = 0
    type: Literal["DailyEM"] = "DailyEM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "EM"
    # NOTE: to override
    
    def calc_alpha(self) -> float:
        res = (
            self.params.alpha
            if self.params.alpha is not None
            else 2 / (self.params.span + 1)
            if self.params.span is not None
            else None
        )
        assert res is not None, self
        return res
    
    @classmethod
    def params_suffix(
        cls,
        span: float | None = None,
        alpha: float | None = None,
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
        return f"{cls.SUFFIX}-" + params

@dataclass(frozen=True)
class DailyChunkedEMD(DailyChunkedEM):
    pass

@dataclass(frozen=True)
class DailyChunkedEMEPParams(DailyEMParams):
    pspan: float | None = None
    palpha: float | None = None

@dataclass(frozen=True)
class DailyChunkedEMEP(DailyChunkedEM):

    @classmethod
    def params_suffix(
        cls,
        span: float | None = None,
        alpha: float | None = None,
        palpha: float | None = None,
        pspan: float | None = None,
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
        if palpha is not None:
            params += f"-{palpha}"
        elif pspan is not None:
            params += f"-{pspan}"
        else:
            raise ValueError(cls)
        return params

    #     span: float | None = None,
    #     alpha: float | None = None,
    #     palpha: float | None = None,
    #     pspan: float | None = None,
    # ):

    #         span=span,
    #         alpha=alpha,
    #         palpha=palpha,
    #         pspan=pspan,

@dataclass(frozen=True)
class DailyChunkedEMPParams(DailyEMParams):
    span: float | None = None
    alpha: float | None = None
    pwindow: int | None = None

@dataclass(frozen=True)
class DailyChunkedEMP(DailyChunkedEM):

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

        # span: float | None = None,
        # alpha: float | None = None,
        # pwindow: int | None = None,

        #     span=span,
        #     alpha=alpha,
        #     pwindow=pwindow,

__all__ = [
    "DailyEMParams",
    "DailyChunkedEM",
    "DailyChunkedEMD",
    "DailyChunkedEMEPParams",
    "DailyChunkedEMEP",
    "DailyChunkedEMPParams",
    "DailyChunkedEMP",
]