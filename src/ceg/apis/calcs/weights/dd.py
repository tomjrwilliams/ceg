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
def calc_rolling_dd(
    v_rx: np.ndarray,
    v_null: np.ndarray,
    v_sample: np.ndarray,
    window: int,
    n_t: int,
    ann: float = 252.0,
):
    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    t = 0
    while t < n_t:
        if not v_sample[t]:
            res[t] = np.nan
            res_null[t] = True
            t += 1
            continue

        elif v_null[t]:
            raise ValueError((t,))

        tt = t - (window - 1)
        if tt < 0:
            tt = 0

        cum_rx = 0
        cum_hwm = 0
        max_dd = 0
        n_rx = 0

        while tt <= t:
            if v_null[tt]:
                tt += 1
                continue

            rx = v_rx[tt]
            cum_rx += rx
            n_rx += 1

            if cum_rx > cum_hwm:
                cum_hwm = cum_rx
                dd = 0
            else:
                dd += rx

            if dd < max_dd:
                max_dd = dd

            tt += 1

        res[t] = -1 * max_dd
        # / (
        #     ((cum_rx) / n_rx) * ann
        # )

        t += 1

    return res


@nb.jit()
def calc_calmar_rolling(
    v_rx: np.ndarray,
    v_null: np.ndarray,
    v_sample: np.ndarray,
    window: int,
    n_t: int,
    ann: float = 252.0,
):
    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    t = 0
    while t < n_t:
        if not v_sample[t]:
            res[t] = np.nan
            res_null[t] = True
            t += 1
            continue

        elif v_null[t]:
            raise ValueError((t,))

        tt = t - (window - 1)
        if tt < 0:
            tt = 0

        cum_rx = 0
        cum_hwm = 0
        max_dd = 0
        n_rx = 0

        while tt <= t:
            if v_null[tt]:
                tt += 1
                continue

            rx = v_rx[tt]
            cum_rx += rx
            n_rx += 1

            if cum_rx > cum_hwm:
                cum_hwm = cum_rx
                dd = 0
            else:
                dd += rx

            if dd < max_dd:
                max_dd = dd

            tt += 1

        if max_dd < 0:
            res[t] = (((cum_rx) / n_rx) * ann) / (
                -1 * max_dd
            )
        else:
            res[t] = np.nan
            res_null[t] = True

        t += 1

    return res, res_null


@nb.jit()
def calc_calmar_ewm(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_sample: np.ndarray,
    window: float,
    n_t: int,
    ann: float = 252.0,
):

    res = np.zeros((n_t,), dtype=float)
    res_null = np.zeros((n_t,), dtype=np.bool)

    alpha = window
    decay = 1 - window

    n_rx = 0
    mu_rx = 0
    mu_rx_sq = 0
    max_rx = 0
    hwm_rx = 0
    cum_rx = 0
    dd_max = 0

    t: int = 0
    while t < n_t:
        if not nd_null[t]:
            rx = nd_rx[t]
            cum_rx += rx
            rx_sq = rx * rx

            if cum_rx > max_rx:
                max_rx = hwm_rx = cum_rx
            else:
                max_rx = (max_rx * decay) + (cum_rx * alpha)

            dd_rx = hwm_rx - cum_rx
            if dd_rx > dd_max:
                dd_max = dd_rx

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
            if dd_max > 0:
                res[t] = (mu_rx * 252.0) / dd_max
            else:
                res[t] = np.nan
                res_null[t] = True

        t += 1

    return res, res_null


__all__ = [
    "calc_rolling_dd",
    "calc_calmar_rolling",
    "calc_calmar_ewm",
]
