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


# @dataclass(frozen=True)
# class DailyMEPParams(DailyMParams):
#     window: int | None = None
#     pspan: float | None = None
#     palpha: float | None = None

#     def build_id(self, cls: Type[DailyM]):
#         window = self.window
#         palpha = self.palpha
#         pspan = self.pspan
#         if window is None:
#             params = "ex"
#         else:
#             params = str(window)
#         if palpha is not None:
#             params += f"-{palpha}"
#         elif pspan is not None:
#             params += f"-{pspan}"
#         else:
#             raise ValueError(cls)
#         return f"{cls.SUFFIX}-" + params


# @dataclass(frozen=True)
# class DailyMEP(DailyM):
#     params: DailyMEPParams

#     @classmethod
#     def new(
#         cls,
#         parent: Daily,
#         window: int | None = None,
#         pspan: float | None = None,
#         palpha: float | None = None,
#         project: Project | None = None,
#     ):
#         params = DailyMEPParams(
#             window=window,
#             pspan=pspan,
#             palpha=palpha,
#         )
#         assert cls.SUFFIX != "M", cls
#         id=parent.extend_id(params.build_id(cls))
#         if project is not None:
#             id=project.extend_id(path)
#         return cls(
#             id=id,
#             parent=parent,
#             params=params,
#             project=project,
#         )


# __all__ = [
#     "DailyMEPParams",
#     "DailyMEP",
# ]
# backward freq of hitting em max, min

# or above a, med, etc.

# decay impulse boolean instead of rolling relative freq
