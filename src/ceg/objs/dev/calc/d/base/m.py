
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

from .root import *

#  ------------------

@dataclass(frozen=True)
class DailyMParams(DailyParams):
    chunk: int = 32
    # TODO: db?

    window: int | None = None

    # NOTE: chunk not in path, as doesnt change value
    # only back-end layout
    # NOTE: likewise for back-end type: np vs pq etc.


@dataclass(frozen=True)
class DailyChunkedM(DailyChunked):
    parent: Object | str
    # tags: DailyTags
    params: DailyMParams

    version: int = 0
    type: Literal["DailyM"] = "DailyM"
    # NOTE: to change

    SUFFIX: ClassVar[str] = "M"
    # NOTE: to override
    
    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
    ):
        if window is None:
            params = "ex"
        else:
            params = str(window)
        return f"{cls.SUFFIX}-" + params

@dataclass(frozen=True)
class DailyChunkedMD(DailyChunkedM):
    pass

@dataclass(frozen=True)
class DailyChunkedMEPParams(DailyMParams):
    window: int | None = None
    pspan: float | None = None
    palpha: float | None = None

@dataclass(frozen=True)
class DailyChunkedMEP(DailyChunkedM):

    @classmethod
    def params_suffix(
        cls,
        window: int | None = None,
        palpha: float | None = None,
        pspan: float | None = None,
    ):
        if window is None:
            params = "ex"
        else:
            params = str(window)
        if palpha is not None:
            params += f"-{palpha}"
        elif pspan is not None:
            params += f"-{pspan}"
        else:
            raise ValueError(cls)
        return f"{cls.SUFFIX}-" + params

        # window: int | None = None,
        # palpha: float | None = None,
        # pspan: float | None = None,

        #     window=window,
        #     palpha=palpha,
        #     pspan=pspan,

@dataclass(frozen=True)
class DailyChunkedMPParams(DailyMParams):
    pwindow: int | None = None

@dataclass(frozen=True)
class DailyChunkedMP(DailyChunkedM):

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

        # window: int | None = None,
        # pwindow: int | None = None,

        #     window=window,
        #     pwindow=pwindow,

#  ------------------

__all__ = [
    "DailyMParams",
    "DailyChunkedM",
    "DailyChunkedMD",
    "DailyChunkedMEPParams",
    "DailyChunkedMEP",
    "DailyChunkedMPParams",
    "DailyChunkedMP",
]