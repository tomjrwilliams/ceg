
from typing import Callable
import polars as pl
import numpy as np
import numba as nb
import math

from ...validation import assert_is

from ...pl import *
from ..utils import *

# ----------------------------------------

@nb.jit()
def smooth_cum(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    n_t: int,
):
    
    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    v_cum = 0
    n_v: int = 0

    t: int = 0
    while t < n_t:

        if nd_null[t]:
            res[t] = np.nan
            res_null[t] = True
            t += 1
            continue

        v = nd_rx[t]

        if np.isnan(v):
            pass
        elif n_v == 0:
            v_cum = v
        else:
            v_cum += v

        res[t] = v_cum
        n_v += 1
        t += 1
    
    return res, res_null

@nb.jit()
def smooth_ewm(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    alpha: float,
    n_t: int,
):
    
    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    decay = 1 - alpha

    v_cum = 0
    n_v: int = 0

    t: int = 0

    v: float = 0

    while t < n_t:

        if np.isnan(nd_rx[t]):
            res[t] = v
            t += 1
            continue

        if n_v == 0:
            v_cum = nd_rx[t]
        else:
            v_cum = (
                (v_cum * decay) + (nd_rx[t] * alpha)
            )

        v = v_cum

        res[t] = v
        n_v += 1
        t += 1
    
    return res, res_null

@nb.jit()
def smooth_rolling(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    window: int,
    n_t: int,
):
    
    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    v: float = 0

    t: int = 0
    while t < n_t:

        if np.isnan(nd_rx[t]):
            res[t] = v
            t += 1
            continue

        tt = t - (window - 1)
        if tt < 0:
            tt = 0

        n_rx = 0
        sum_rx = 0

        # TODO: back count off the sum as we roll forward
        # so only one pass is needed
        # need second t iterators (for back end, step up when we go next - eg. null etc.)

        while tt <= t:
            if np.isnan(nd_rx[tt]):
                tt += 1
                continue
            
            rx = nd_rx[tt]

            sum_rx += rx
            n_rx += 1
                
            tt += 1

        if n_rx > 1:
            v = sum_rx / n_rx

        res[t] = v

        t += 1
    
    return res, res_null

__all__ = [
    "smooth_cum",
    "smooth_ewm",
    "smooth_rolling",
]