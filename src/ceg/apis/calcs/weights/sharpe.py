from typing import Callable
import polars as pl
import numpy as np
import numba as nb
import math

from ...validation import assert_is

from ...pl import *
from ..utils import *

# ----------------------------------------

# TODO: sharpe calc on residual of each rx to equal weighted basket?


@nb.jit()
def calc_sharpe_rolling(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_sample: np.ndarray,
    window: int,
    n_t: int,
    ann: float = 252.0,
):

    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    t: int = 0
    l: int = 0

    n_rx = 0
    sum_rx = 0
    sum_rx_sq = 0

    while t < n_t:
        if not nd_null[t]:
            rx = nd_rx[t]

            sum_rx += rx
            sum_rx_sq += rx * rx
            n_rx += 1

        if not nd_sample[t]:
            res[t] = np.nan
            res_null[t] = True
            t += 1
            continue

        # elif nd_null[t]:
        #     raise ValueError((t,))

        if window > 0:
            l_new = t - (window - 1)
            if l_new < 0:
                l_new = 0

            while l_new > l:
                if nd_null[l]:
                    l += 1
                    continue
                rx = nd_rx[l]
                sum_rx -= rx
                sum_rx_sq -= rx * rx
                n_rx -= 1

                l += 1

        if n_rx > 1:
            mu_rx = sum_rx / n_rx
            rms_rx = math.sqrt(sum_rx_sq / n_rx)

            if rms_rx == 0:
                res[t] = np.nan
                res_null[t] = True
            else:
                res[t] = (mu_rx * ann) / (
                    rms_rx * math.sqrt(ann)
                )
        else:
            res[t] = np.nan
            res_null[t] = True

        t += 1

    return res, res_null


@nb.jit()
def calc_sharpe_ewm(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_sample: np.ndarray,
    window: float,
    n_t: int,
    ann: float = 252.0,
):

    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    decay = 1 - window

    n_rx = 0
    mu_rx = 0
    mu_rx_sq = 0

    t: int = 0
    while t < n_t:
        if not nd_null[t]:
            rx = nd_rx[t]
            rx_sq = rx * rx

            if n_rx == 0:
                mu_rx = rx
                mu_rx_sq = rx * rx
            else:
                mu_rx = (mu_rx * decay) + (rx * window)
                mu_rx_sq = (mu_rx_sq * decay) + (
                    rx_sq * window
                )

            n_rx += 1

        if nd_sample[t] and n_rx > 1:
            if mu_rx_sq > 0:
                res[t] = (mu_rx * ann) / (
                    math.sqrt(mu_rx_sq) * math.sqrt(ann)
                )
            else:
                res[t] = np.nan
                res_null[t] = True

        t += 1

    return res, res_null


__all__ = [
    "calc_sharpe_rolling",
    "calc_sharpe_ewm",
]
