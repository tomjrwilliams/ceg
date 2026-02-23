
from typing import Callable
import polars as pl
import numpy as np
import numba as nb
import math

from ...validation import assert_is

from ..utils import *

# ----------------------------------------

@nb.jit()
def calc_rx_grid_update(
    rx_cum: float,
    # rx_hwm: float,
    # rx_lwm: float,
    rx_grid_chg: int,
    rx_grid_last: float,
    rx_grid_up: float,
    rx_grid_dn: float,
    rx_grid_width: float,
    re_grid: bool,
):
    if rx_grid_chg == 0:
        raise ValueError(rx_grid_chg)
    
    if re_grid:
        rx_grid_last = rx_cum
    elif rx_grid_chg > 0:
        rx_grid_last = rx_grid_up + (
            (rx_grid_chg - 1) * rx_grid_width
        )
    elif rx_grid_chg < 0:
        rx_grid_last = rx_grid_dn - (
            (rx_grid_chg + 1) * rx_grid_width
        )
    else:
        raise ValueError((rx_grid_chg, re_grid))

    rx_grid_up = rx_grid_last + rx_grid_width
    rx_grid_dn = rx_grid_last - rx_grid_width

    # if rx_cum != rx_hwm:
    #     rx_grid_top = 0
    #     rx_grid_top_v = rx_grid_up
    #     while rx_grid_top_v < rx_hwm:
    #         rx_grid_top += 1
    #         rx_grid_top_v += rx_grid_width

    # if rx_cum != rx_lwm:
    #     rx_grid_btm = 0
    #     rx_grid_btm_v = rx_grid_dn
    #     while rx_grid_btm_v > rx_lwm:
    #         rx_grid_btm += 1
    #         rx_grid_btm_v -= rx_grid_width

    return (
        rx_grid_last,
        rx_grid_up,
        rx_grid_dn,
        # rx_grid_top,
        # rx_grid_btm,
    )

@nb.jit()
def calc_rx_grid_chg(
    rx_cum: float,
    rx_grid_up: float,
    rx_grid_dn: float,
    rx_grid_width: float,
):
    
    if rx_cum >= rx_grid_up:

        rx_grid_chg = 1
        while rx_cum >= rx_grid_up:
            rx_grid_up += rx_grid_width
            rx_grid_chg += 1

        return rx_grid_chg

    elif rx_cum <= rx_grid_dn:

        rx_grid_chg = -1
        while rx_cum <= rx_grid_dn:
            rx_grid_dn -= rx_grid_width
            rx_grid_chg -= 1

        return rx_grid_chg
        
    else:
        return 0

@nb.jit()
def calc_metrics_grid(
    n_t: int,
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    rx_grid_width: float,
    re_grid: bool,
):

    nd_rx_grid_len = np.zeros((n_t,))
    nd_rx_grid_chg = np.zeros((n_t,))
    nd_rx_grid_top = np.zeros((n_t,))
    nd_rx_grid_btm = np.zeros((n_t,))

    nd_rx_grid_net = np.zeros((n_t,))
    nd_rx_grid_grs = np.zeros((n_t,))

    # return the up / down thresholds?

    rx_grid_last = 0

    rx_grid_len = 0
    rx_grid_net = 0
    rx_grid_grs = 0

    rx_grid_up = rx_grid_last + rx_grid_width
    rx_grid_dn = rx_grid_last - rx_grid_width

    rx_grid_top = 0
    rx_grid_btm = 0

    start =-1
    rx_cum = 0

    t = 0
    while t < n_t:
        if nd_null[t]:

            nd_rx_grid_len[t] = np.nan
            nd_rx_grid_chg[t] = np.nan
            nd_rx_grid_top[t] = np.nan
            nd_rx_grid_btm[t] = np.nan

            nd_rx_grid_net[t] = np.nan
            nd_rx_grid_grs[t] = np.nan
            
            t += 1
            continue

        if start == -1:
            start = t

        rx = nd_rx[t]
        rx_cum += rx

        rx_grid_len += 1

        is_grid_chg = False

        rx_grid_chg = calc_rx_grid_chg(
            rx_cum,
            rx_grid_up,
            rx_grid_dn,
            rx_grid_width,
        )

        rx_grid_len += 1
        rx_grid_net += rx
        rx_grid_grs += abs(rx)

        if rx_grid_chg != 0:
            is_grid_chg = True

            nd_rx_grid_len[t] = rx_grid_len
            nd_rx_grid_chg[t] = rx_grid_chg
            nd_rx_grid_net[t] = rx_grid_net
            nd_rx_grid_grs[t] = rx_grid_grs

            (
                rx_grid_last,
                rx_grid_up,
                rx_grid_dn,
                # rx_grid_top,
                # rx_grid_btm,
            ) = calc_rx_grid_update(
                rx_cum,
                # rx_hwm,
                # rx_lwm,
                rx_grid_chg,
                rx_grid_last,
                rx_grid_up,
                rx_grid_dn,
                rx_grid_width,
                re_grid,
            )

            rx_grid_len = 0
            rx_grid_net = 0
            rx_grid_grs = 0
        else:
            nd_rx_grid_len[t] = np.nan
            nd_rx_grid_chg[t] = np.nan

        # nd_rx_grid_top[t] = rx_grid_top
        # nd_rx_grid_btm[t] = rx_grid_btm

        t += 1

    return (
        nd_rx_grid_len,
        nd_rx_grid_chg,
        nd_rx_grid_top,
        nd_rx_grid_btm,
        nd_rx_grid_net,
        nd_rx_grid_grs,
    )
    
@nb.jit()
def basket_metrics_grid(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_all_null: np.ndarray,
    rx_grid_width: float,
    re_grid: bool,
):
    nd_rx_grid_len = np.zeros(nd_rx.shape)
    nd_rx_grid_chg = np.zeros(nd_rx.shape)
    nd_rx_grid_top = np.zeros(nd_rx.shape)
    nd_rx_grid_btm = np.zeros(nd_rx.shape)

    nd_rx_grid_net = np.zeros(nd_rx.shape)
    nd_rx_grid_grs = np.zeros(nd_rx.shape)

    n_x = nd_rx.shape[1]
    n_t = nd_rx.shape[0]

    x = 0
    while x < n_x:
        (
            nd_rx_grid_len[:, x],
            nd_rx_grid_chg[:, x],
            nd_rx_grid_top[:, x],
            nd_rx_grid_btm[:, x],
            nd_rx_grid_net[:, x],
            nd_rx_grid_grs[:, x],
        ) = calc_metrics_grid(
            n_t,
            nd_rx[:, x],
            nd_null[:, x],
            rx_grid_width,
            re_grid,
        )
        x += 1

    return (
        nd_rx_grid_len,
        nd_rx_grid_chg,
        nd_rx_grid_top,
        nd_rx_grid_btm,
        nd_rx_grid_net,
        nd_rx_grid_grs,
    )

def metrics_grid_df(
    df: pl.DataFrame,
    rx_grid_width: float,
    re_grid: bool,
):
    (
        nd_rx, nd_null, nd_active,v_null_expr, all_null
    ) = cols_with_null_map(
        df, drop = ["date"]
    )
    (
        nd_rx_grid_len,
        nd_rx_grid_chg,
        nd_rx_grid_top,
        nd_rx_grid_btm,
        nd_rx_grid_net,
        nd_rx_grid_grs,
    ) = basket_metrics_grid(
        nd_rx,
        nd_null,
        all_null,
        rx_grid_width,
        bool(re_grid),
    )
    
    return df.pipe(
        repack_to_df,
        f=None,
        pipe=None,
        null=None,
        **{
            # drop rx / grid?
            "rx_grid_len": nd_rx_grid_len,
            "rx_grid_chg": nd_rx_grid_chg,
            "rx_grid_top": nd_rx_grid_top,
            "rx_grid_btm": nd_rx_grid_btm,
            "rx_grid_net": nd_rx_grid_net,
            "rx_grid_grs": nd_rx_grid_grs,
        },
    )

# ----------------------------------------

__all__ = [
    "metrics_grid_df",
]