from typing import TypeAlias
import math

from .events import *
from .snaps import *
from .values import *
from .depends import *
from .objects import *
from .storage import *
from .handlers import *
from .daily import *
from .collections import *

from . import handlers
from . import queries

from ..apis import t, kv, assert_is

from frozendict import frozendict
frozend: TypeAlias = frozendict

def fdict(**kwargs):
    return frozendict(**kwargs)

NEW_LINE = "\n"


import numpy as np
import polars as pl

def col_as_np(df: pl.DataFrame, key: str):
    return df.get_column(key).to_numpy()


def window_days(window: str):
    unit = window[-1].lower()
    n = int(window[:-1])
    unit = (
        1 if unit == "d"
        else 7 if unit == "w"
        else 30 if unit == "m"
        else 365 if unit == "y"
        else None
    )
    if unit is None:
        raise ValueError(window)
    return n * unit

def norm_expr(
    e: pl.Expr,
    norm: str,
    n: int,
    diff: bool = False,
):
    if diff:
        e = e.diff()
    
    func, param = norm.split("-", maxsplit=1)

    i = pl.int_range(0, n)

    if func.startswith("ew"):
        ann = 1
        if "-" in param:
            param, ann = param.split("-")
            ann = int(ann)
        span = int(param)
        alpha = 2 / (span + 1)
        min_samples = max(8, span / 64)

    elif func.startswith("win"):
        ann = 1
        if "-" in param:
            param, ann = param.split("-")
            ann = int(ann)
        window = int(param)
        min_samples = max(8, window / 64)

    if func == "ewrms":
        ann = math.sqrt(ann)
        e_global = e.pow(2).mean().sqrt().mul(ann)
        e_warm = pl.when(i < min_samples).then(e_global).otherwise(e)
        return (
            e_warm.pow(2)
            .ewm_mean(
                alpha=alpha,
                min_samples=1, 
                ignore_nulls=True,
            )
            .sqrt()
            .mul(ann)
        )
    elif func == "ewmean":
        e_global = e.mean().mul(ann)
        e_warm = pl.when(i < min_samples).then(e_global).otherwise(e)
        return (
            e_warm.ewm_mean(
                alpha=alpha,
                min_samples=1,
                ignore_nulls=True,
            )
            .mul(ann)
        )
    elif func == "winmean":
        e_global = e.mean().mul(ann)
        e_warm = pl.when(i < min_samples).then(e_global).otherwise(e)
        return (
            e_warm.rolling_mean(
                window,
                min_samples=1, 
            )
            .mul(ann)
        )
    raise ValueError(norm)

from functools import lru_cache
import itertools

@lru_cache(maxsize=12)
def rank_permutations(n: int):
    return list(sorted(itertools.permutations(list(range(n)), n)))

@lru_cache(maxsize=12)
def rank_combinations(nl: int, nr: int):
    return list(itertools.product(list(range(nl)), list(range(nr))))
