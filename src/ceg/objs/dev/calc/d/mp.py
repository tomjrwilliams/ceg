
from __future__ import annotations
import os
from typing import (
    TYPE_CHECKING,
    Type,
    TypeVar,
    ClassVar,
    Literal,
    Protocol,
    ParamSpec,
    Callable,
    Concatenate,
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

from ..... import apis
from .....apis import t
from .....apis import kv
from .....engine import *

from .m import *

@dataclass(frozen=True)
class DailyMPParams(DailyMParams):
    pwindow: int | None = None

@dataclass(frozen=True)
class DailyMP(DailyM):

    params: DailyMPParams

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

    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        window: int | None = None,
        pwindow: int | None = None,
    ): ...
    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ],
        window: int | None = None,
        pwindow: int | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ): ...
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: Callable[
            Concatenate[EngineRoot, P], O
        ] | None = None,
        window: int | None = None,
        pwindow: int | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                window=window,
            )),
            parent=parent,
            params=DailyMPParams(
                window=window,
                pwindow=pwindow,
            ),
        )
__all__ = [
    "DailyMPParams",
    "DailyMP",
]
# backward freq of hitting em max, min

# or above a, med, etc.