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

    def build_id(self, cls: Type[DailyEM]):
        span = self.span
        alpha = self.alpha
        pwindow = self.pwindow
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


@dataclass(frozen=True)
class DailyEMP(DailyEM):
    params: DailyEMPParams

    @classmethod
    def new(
        cls,
        parent: Daily,
        span: float | None = None,
        alpha: float | None = None,
        pwindow: int | None = None,
        project: Project | None = None,
    ):
        params = DailyEMPParams(
            span=span,
            alpha=alpha,
            pwindow=pwindow,
        )
        assert cls.SUFFIX != "M", cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )


__all__ = [
    "DailyEMPParams",
    "DailyEMP",
]
# backward freq of hitting em max, min

# or above a, med, etc.

# rolling relative freq

# on exp max, min, etc.
