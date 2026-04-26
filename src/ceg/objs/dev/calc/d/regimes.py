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
from typing import overload, Literal
import numpy as np
import polars as pl
import numba as nb

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
from .....apis import assert_is, t, kv
from .....apis.calcs.regimes.dd import with_dd_regime
from .....engine import *


@dataclass(frozen=True)
class DailyDDRegimeParams(DailyParams):
    chunk: int = 32

    threshold: float = 1.0
    span: int = 16
    window: int = 16
    continuation: float = 0.1

    # TODO: cum

    def build_id(
        self,
        cls: Type[DailyDDRegime],
    ):
        threshold = self.threshold
        span = self.span
        window = self.window
        continuation = self.continuation
        return f"ddreg-{span}-{window}-{threshold}-{continuation}"


P = ParamSpec("P")


@dataclass(frozen=True)
class DailyDDRegime(DailyWithDailyParent, DailyChunked):
    parent: Object | str
    # tags: DailyTags

    params: DailyDDRegimeParams

    project: Project | None = None

    version: int = 0
    type: Literal["DailyDDRegime"] = "DailyDDRegime"
    # NOTE: to change

    @dataclass(frozen=True)
    class Snap(Daily.Snap):
        type: Literal["DailyDDRegimeSnap"] = (
            "DailyDDRegimeSnap"
        )

    @classmethod
    def new(
        cls,
        parent: Daily,
        threshold: float = 1.0,
        span: int = 16,
        window: int = 16,
        continuation: float = 0.0,
        # TODO: cum: bool = True
        project: Project | None = None,
    ) -> DailyDDRegime:
        params = DailyDDRegimeParams(
            threshold=threshold,
            span=span,
            window=window,
            # continuation=continuation,
        )
        assert continuation == 0, cls
        id=parent.extend_id(params.build_id(cls))
        if project is not None:
            id=project.extend_id(path)
        return cls(
            id=id,
            parent=parent,
            params=params,
            project=project,
        )

    def handler_object_new(self) -> HandleNewDailyDDRegime:
        return HandleNewDailyDDRegime()


@dataclass(frozen=True)
class HandleNewDailyDDRegime(
    HandleNewDailyFromDailyParentPl
):
    def calc_df(
        self,
        df: pl.DataFrame,
        event: ObjectChange,
        obj: DailyDDRegime,
    ) -> pl.DataFrame:
        params = obj.params
        return df.pipe(
            with_dd_regime,
            rx=VALUE,
            span=params.span,
            threshold=params.threshold,
            window=params.window,
            continuation=params.continuation,
        ).select("date", pl.col("label").alias(VALUE))


__all__ = [
    "DailyDDRegimeParams",
    "DailyDDRegime",
    "with_dd_regime",
]
