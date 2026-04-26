from typing import Callable
import polars as pl
import numpy as np
import numba as nb
import math

from ...validation import assert_is

from ..utils import *

# ----------------------------------------


@nb.jit()
def calc_metrics_ewm(
    n_t: int,
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    span: float,
):
    nd_rx_mu = np.zeros((n_t,))
    nd_rx_cum = np.zeros((n_t,))
    nd_rx_cmu = np.zeros((n_t,))
    nd_rx_max = np.zeros((n_t,))
    nd_rx_min = np.zeros((n_t,))
    nd_rx_hwm = np.zeros((n_t,))
    nd_rx_lwm = np.zeros((n_t,))

    nd_is_hwm = np.zeros((n_t,))
    nd_is_lwm = np.zeros((n_t,))

    nd_hwm_len = np.zeros((n_t,))
    nd_lwm_len = np.zeros((n_t,))

    nd_hwm_net = np.zeros((n_t,))
    nd_hwm_grs = np.zeros((n_t,))

    nd_lwm_net = np.zeros((n_t,))
    nd_lwm_grs = np.zeros((n_t,))

    nd_osc_15 = np.zeros((n_t,))
    nd_osc_30 = np.zeros((n_t,))

    # return the up / down thresholds?

    rx_mu = np.nan
    rx_cum = 0
    rx_cmu = np.nan
    rx_max = np.nan
    rx_min = np.nan
    rx_hwm = np.nan
    rx_lwm = np.nan

    rx_ms = np.nan
    rx_rms = np.nan

    alpha = 2 / (span + 1)
    decay = 1 - alpha

    alpha_15 = 2 / ((span * 1.5) + 1)
    decay_15 = 1 - alpha_15

    alpha_30 = 2 / ((span * 3) + 1)
    decay_30 = 1 - alpha_30

    rx_cmu_15 = np.nan
    rx_cmu_30 = np.nan

    rx_osc_15 = np.nan
    rx_osc_30 = np.nan

    hwm_len = 0
    hwm_net = 0
    hwm_grs = 0

    lwm_len = 0
    lwm_net = 0
    lwm_grs = 0

    t = 0
    while t < n_t:
        if nd_null[t]:
            nd_rx_mu[t] = np.nan
            nd_rx_cum[t] = np.nan
            nd_rx_cmu[t] = np.nan
            nd_rx_max[t] = np.nan
            nd_rx_min[t] = np.nan
            nd_rx_hwm[t] = np.nan
            nd_rx_lwm[t] = np.nan

            nd_is_hwm[t] = np.nan
            nd_is_lwm[t] = np.nan
            nd_hwm_len[t] = np.nan
            nd_lwm_len[t] = np.nan
            nd_hwm_net[t] = np.nan
            nd_hwm_grs[t] = np.nan
            nd_lwm_net[t] = np.nan
            nd_lwm_grs[t] = np.nan

            nd_osc_15[t] = np.nan
            nd_osc_30[t] = np.nan

            t += 1
            continue

        rx = nd_rx[t]
        rx_sq = rx * rx
        rx_cum += rx

        is_hwm = False
        is_lwm = False

        if np.isnan(rx_mu):
            rx_cmu = rx_cmu_15 = rx_cmu_30 = rx_cum
            rx_mu = rx
            if rx_cum > 0:
                rx_max = rx_cum
                rx_min = 0
            else:
                rx_min = rx_cum
                rx_max = 0

            rx_hwm = rx_max
            rx_lwm = rx_min

            rx_ms = rx_sq
            rx_rms = abs(rx)

        else:
            rx_mu = (rx_mu * decay) + (rx * alpha)
            rx_cmu = (rx_cmu * decay) + (rx_cum * alpha)
            rx_cmu_15 = (rx_cmu_15 * decay_15) + (
                rx_cum * alpha_15
            )
            rx_cmu_30 = (rx_cmu_30 * decay_30) + (
                rx_cum * alpha_30
            )
            rx_ms = (rx_ms * decay) + (rx_sq * alpha)

            rx_rms = math.sqrt(rx_ms)

            if rx_cum > rx_max:  # ge?
                rx_max = rx_cum
                rx_hwm = rx_max
                is_hwm = True
                # TODO: bool signal for is_hwm

                rx_grid_top = 0
            else:
                rx_max = (rx_max * decay) + (rx_cmu * alpha)

            if rx_cum < rx_min:  # le?
                rx_min = rx_cum
                rx_lwm = rx_min
                is_lwm = True
                # TODO: bool signal for is_lwm

                rx_grid_btm = 0
            else:
                rx_min = (rx_min * decay) + (rx_cmu * alpha)

        hwm_len += 1
        hwm_net += rx
        hwm_grs += abs(rx)

        lwm_len += 1
        lwm_net += rx
        lwm_grs += abs(rx)

        nd_hwm_len[t] = hwm_len
        nd_hwm_net[t] = hwm_net
        nd_hwm_grs[t] = hwm_grs

        nd_lwm_len[t] = lwm_len
        nd_lwm_net[t] = lwm_net
        nd_lwm_grs[t] = lwm_grs

        if is_hwm:
            nd_is_hwm[t] = True
            hwm_len = 0
            hwm_net = 0
            hwm_grs = 0
        else:
            nd_is_hwm[t] = np.nan

        if is_lwm:
            nd_is_lwm[t] = True
            lwm_len = 0
            lwm_net = 0
            lwm_grs = 0
        else:
            nd_is_lwm[t] = np.nan

        rx_osc_15 = rx_cmu - rx_cmu_15
        rx_osc_30 = rx_cmu - rx_cmu_30

        nd_rx_mu[t] = rx_mu
        nd_rx_cum[t] = rx_cum
        nd_rx_cmu[t] = rx_cmu
        nd_rx_max[t] = rx_max
        nd_rx_min[t] = rx_min
        nd_rx_hwm[t] = rx_hwm
        nd_rx_lwm[t] = rx_lwm

        nd_osc_15[t] = rx_osc_15
        nd_osc_30[t] = rx_osc_30

        t += 1

    return (
        nd_rx_mu,
        nd_rx_cum,
        nd_rx_cmu,
        nd_rx_max,
        nd_rx_min,
        nd_rx_hwm,
        nd_rx_lwm,
        nd_is_hwm,
        nd_is_lwm,
        nd_hwm_len,
        nd_lwm_len,
        nd_hwm_net,
        nd_hwm_grs,
        nd_lwm_net,
        nd_lwm_grs,
        nd_osc_15,
        nd_osc_30,
    )


@nb.jit()
def basket_metrics_ewm(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_all_null: np.ndarray,
    span: float,
):
    nd_rx_mu = np.zeros(nd_rx.shape)
    nd_rx_cum = np.zeros(nd_rx.shape)
    nd_rx_cmu = np.zeros(nd_rx.shape)
    nd_rx_max = np.zeros(nd_rx.shape)
    nd_rx_min = np.zeros(nd_rx.shape)
    nd_rx_hwm = np.zeros(nd_rx.shape)
    nd_rx_lwm = np.zeros(nd_rx.shape)

    nd_is_hwm = np.zeros(nd_rx.shape)
    nd_is_lwm = np.zeros(nd_rx.shape)
    nd_hwm_len = np.zeros(nd_rx.shape)
    nd_lwm_len = np.zeros(nd_rx.shape)
    nd_hwm_net = np.zeros(nd_rx.shape)
    nd_hwm_grs = np.zeros(nd_rx.shape)
    nd_lwm_net = np.zeros(nd_rx.shape)
    nd_lwm_grs = np.zeros(nd_rx.shape)

    nd_osc_15 = np.zeros(nd_rx.shape)
    nd_osc_30 = np.zeros(nd_rx.shape)

    n_x = nd_rx.shape[1]
    n_t = nd_rx.shape[0]

    x = 0
    while x < n_x:
        # print("calc", x)
        (
            nd_rx_mu[:, x],
            nd_rx_cum[:, x],
            nd_rx_cmu[:, x],
            nd_rx_max[:, x],
            nd_rx_min[:, x],
            nd_rx_hwm[:, x],
            nd_rx_lwm[:, x],
            nd_is_hwm[:, x],
            nd_is_lwm[:, x],
            nd_hwm_len[:, x],
            nd_lwm_len[:, x],
            nd_hwm_net[:, x],
            nd_hwm_grs[:, x],
            nd_lwm_net[:, x],
            nd_lwm_grs[:, x],
            nd_osc_15[:, x],
            nd_osc_30[:, x],
        ) = calc_metrics_ewm(
            n_t,
            nd_rx[:, x],
            nd_null[:, x],
            span,
        )
        x += 1

    return (
        nd_rx_mu,
        nd_rx_cum,
        nd_rx_cmu,
        nd_rx_max,
        nd_rx_min,
        nd_rx_hwm,
        nd_rx_lwm,
        nd_is_hwm,
        nd_is_lwm,
        nd_hwm_len,
        nd_lwm_len,
        nd_hwm_net,
        nd_hwm_grs,
        nd_lwm_net,
        nd_lwm_grs,
        nd_osc_15,
        nd_osc_30,
    )


def metrics_ewm_df(
    df: pl.DataFrame,
    span: float,
):
    (nd_rx, nd_null, nd_active, v_null_expr, all_null) = (
        cols_with_null_map(df, drop=["date"])
    )
    print(("calc metrics ewm", span))
    (
        rx_mu,
        rx_cum,
        rx_cmu,
        rx_max,
        rx_min,
        rx_hwm,
        rx_lwm,
        nd_is_hwm,
        nd_is_lwm,
        nd_hwm_len,
        nd_lwm_len,
        nd_hwm_net,
        nd_hwm_grs,
        nd_lwm_net,
        nd_lwm_grs,
        nd_osc_15,
        nd_osc_30,
    ) = basket_metrics_ewm(
        nd_rx,
        nd_null,
        all_null,
        span,
    )
    return df.pipe(
        repack_to_df,
        f=None,
        pipe=None,
        null=None,
        **{
            "rx": (nd_rx, None, None, None, {}),
            "rx_mu": rx_mu,
            "rx_cum": rx_cum,
            "rx_cmu": rx_cmu,
            "rx_max": rx_max,
            "rx_min": rx_min,
            "rx_hwm": rx_hwm,
            "rx_lwm": rx_lwm,
            "is_hwm": nd_is_hwm,
            "is_lwm": nd_is_lwm,
            "hwm_len": nd_hwm_len,
            "lwm_len": nd_lwm_len,
            "hwm_net": nd_hwm_net,
            "hwm_grs": nd_hwm_grs,
            "lwm_net": nd_lwm_net,
            "lwm_grs": nd_lwm_grs,
            "osc_15": nd_osc_15,
            "osc_30": nd_osc_30,
        },
    )


__all__ = [
    "metrics_ewm_df",
    "calc_metrics_ewm",
]
