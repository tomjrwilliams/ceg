
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

from .em import *

@dataclass(frozen=True)
class DailyEMPParams(DailyEMParams):
    span: float | None = None
    alpha: float | None = None
    pwindow: int | None = None

@dataclass(frozen=True)
class DailyEMP(DailyEM):

    params: DailyEMPParams

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

    @overload
    @classmethod
    def new(
        cls,
        parent: EngineRoot | Object,
        t: None = None,
        span: float | None = None,
        alpha: float | None = None,
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
        span: float | None = None,
        alpha: float | None = None,
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
        span: float | None = None,
        alpha: float | None = None,
        pwindow: int | None = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if isinstance(parent, EngineRoot):
            assert t is not None, t
            parent = t(parent, *args, **kwargs)
        return cls(
            path=parent.extend_path(cls.params_suffix(
                span=span,
                alpha=alpha,
            )),
            parent=parent,
            params=DailyEMPParams(
                span=span,
                alpha=alpha,
                pwindow=pwindow,
            ),
        )
__all__ = [
    "DailyEMPParams",
    "DailyEMP",
]
# backward freq of hitting em max, min

# or above a, med, etc.

# rolling relative freq

# on exp max, min, etc.