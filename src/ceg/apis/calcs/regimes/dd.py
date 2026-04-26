import polars as pl
import numpy as np
import numba as nb

from ..utils import *

CACHE = {}


def with_dd_regime(
    df: pl.DataFrame,
    rx: str,
    threshold: float,
    span: int,
    window: int | None = None,
    continuation: float | None = None,
    alias: str = "regime",
    dd: str = "dd",
    hwm: str = "hwm",
    label: str = "label",
    key: str | None = None,
):
    if key is not None and key in CACHE:
        return CACHE[key]

    alpha = 2 / (span + 1)

    s = df.get_column(rx).fill_nan(None)
    is_null = s.is_null().to_numpy()

    s = s.fill_nan(None).cum_sum()
    v = s.fill_null(0).to_numpy()

    res = calc_hwm(v, is_null, alpha=alpha)

    df_ = (
        df.with_columns(pl.Series(name=hwm, values=res))
        .with_columns(
            pl.when(is_null)
            .then(None)
            .otherwise(hwm)
            .alias(hwm)
        )
        .with_columns(
            # null mask?
            pl.col(rx)
            .fill_nan(None)
            .fill_null(0)
            .cum_sum()
            .sub(pl.col(hwm))
            .alias(dd)
        )
    )

    v_rx = df_.get_column(rx).fill_null(0).to_numpy()
    v_is_null = is_null
    v_hwm = df_.get_column(hwm).to_numpy()
    v_dd = df_.get_column(dd).to_numpy()

    regime, labels = calc_dd_regime(
        v_rx,
        v_is_null,
        v_hwm,
        v_dd,
        -1 * threshold,
        continuation=-1 * (continuation or (threshold / 8)),
        window=window or int(span / 8),
    )

    df = (
        df.with_columns(
            pl.Series(name=dd, values=regime).fill_nan(
                None
            ),
            pl.Series(name=label, values=labels).fill_nan(
                None
            ),
        )
        .with_columns(
            pl.col(dd)
            .is_not_null()
            .cast(pl.Int32)
            .reverse()
            .cum_sum()
            .reverse()
            .sub(1)
            .alias(alias),
            pl.col(dd).alias(dd),
        )
        .with_columns(pl.col(alias).max() - pl.col(alias))
    )
    if key is not None:
        CACHE[key] = df
    return df


def calc_dd_regime(
    v_rx: np.ndarray,
    v_is_null: np.ndarray,
    v_hwm: np.ndarray,
    v_dd: np.ndarray,
    threshold: float,
    continuation: float,
    window: int,  # int(span/16): 3y -> 3m
    merge: bool = True,
):
    t = len(v_rx)

    values = np.zeros((t,)) * np.nan
    labels = np.zeros((t,)) * np.nan

    in_dd: bool = False
    peak: int = 0
    trough: int = 0
    next_peak: int = 0

    last_bound: int = 0

    for i in range(t):
        if v_is_null[i]:
            continue

        if i > 0 and (i <= trough or i <= next_peak):
            continue

        # TODO: null treatment of dd?
        # eg. if ends on null, we'll skip?

        dd = v_dd[i]

        if dd < threshold:
            values[i] = 0
            labels[i] = -0.5

            for j in range(i - 1):
                i_prev = i - (j + 1)
                if v_dd[i_prev] == 0:
                    peak = i_prev
                    break

            if not merge:
                if next_peak > 0 and next_peak != peak:
                    rx_acc = 0
                    for j in range(
                        trough + 1, next_peak + 1
                    ):
                        if v_is_null[j]:
                            continue
                        rx_acc += v_rx[j]

                    trough = next_peak
                    values[next_peak] = rx_acc
                    labels[next_peak] = 1

            if peak > 0:
                rx_peak = 0.0
                for j in range(
                    0 if trough == 0 else trough + 1,
                    peak + 1,
                ):
                    if v_is_null[j]:
                        continue
                    rx_peak += v_rx[j]

                values[peak] = rx_peak
                labels[peak] = 1

            trough = i
            trough_v = 0
            dd_j = 0
            since_trough = 0

            for j in range(i + 1, t):
                if v_is_null[j]:
                    continue

                if since_trough == window:
                    values[j] = 0
                    labels[j] = 0.5
                    break

                dd_j += v_rx[j]

                if dd_j < trough_v + continuation:
                    since_trough = 0
                    trough = j
                    trough_v = dd_j
                else:
                    since_trough += 1

                if j == t - 1:
                    trough = t - 1

            rx_dd = 0
            for j in range(
                peak if peak == 0 else peak + 1, trough + 1
            ):
                if v_is_null[j]:
                    continue
                rx_dd += v_rx[j]

            values[trough] = rx_dd
            labels[trough] = -1

            next_peak = t - 1
            for j in range(trough + 1, t):
                if v_dd[j] == 0:
                    next_peak = j
                    break

    if labels[-1] != -1:
        peak = t - 1

        rx_peak = 0
        for j in range(
            trough if trough == 0 else trough + 1, peak + 1
        ):
            if v_is_null[j]:
                continue
            rx_peak += v_rx[j]

        values[peak] = rx_peak
        labels[peak] = 1

    return values, labels


@nb.jit()
def adjust_weight_start(
    v_i_max: np.ndarray,
    v_cond: np.ndarray,
    share: bool,
    n_x: int,
    span: int,
    dd_incr: float,
):
    # max_i_sd = np.std(v_i_max)
    # max_i_sd_sum += max_i_sd # type: ignore
    # max_i_sd_n += 1

    # max_i_sd_mu = max_i_sd_sum / max_i_sd_n

    # if max_i_sd <= 1:
    #     dd_weight = 2.5
    # elif max_i_sd < max_i_sd_mu:
    #     dd_weight = 1.5
    # elif max_i_sd >= max_i_sd_mu:
    #     dd_weight = 1
    # else:
    #     raise ValueError(max_i_sd)

    dd_weight = 1.0
    if share:
        latest_max = np.nan
        n_in_dd = 0
        x = 0
        while x < n_x:
            if v_cond[x]:
                i_max = v_i_max[x]
                n_in_dd += 1
                if np.isnan(latest_max):
                    latest_max = i_max
                elif i_max > latest_max:
                    latest_max = i_max
            x += 1
        if n_in_dd > 1:
            x = 0
            while x < n_x:
                if v_cond[x]:
                    i_max_adj = latest_max - v_i_max[x]
                    if i_max_adj > span:
                        i_max_adj = span
                    dd_weight += dd_incr * (
                        1 - (i_max_adj / span)
                    )
                x += 1
    return dd_weight


@nb.jit()
def basket_dd_regime(
    v_rx: np.ndarray,
    v_is_null: np.ndarray,
    v_null: np.ndarray,
    threshold: float,
    span: int,
    window: int,  # int(span/16): 3y -> 3m
    continuation: float,
    merge: bool = True,
    threshold_share: bool = False,
    recovery_share: bool = False,
    continuous: bool = False,  # positioning
    dd_speedup_max: float = 2 / 3,
    dd_speedup_threshold: float = 1 / 3,
):
    assert continuation >= 0, continuation
    assert threshold > 0, threshold

    alpha = 2 / (span + 1)
    decay = 1 - alpha

    allocs = np.zeros(v_rx.shape) * np.nan
    rx_cums = np.zeros(v_rx.shape) * np.nan
    labels = np.zeros(v_rx.shape) * np.nan

    n_t: int = v_rx.shape[0]
    n_x: int = v_rx.shape[1]

    # n_t = assert_is(n_t, v_null.shape[0])
    # n_x = assert_is(n_x, v_is_null.shape[1])

    dd_speedup_threshold = threshold * dd_speedup_threshold
    dd_speedup_max = threshold * dd_speedup_max
    dd_speedup_rate = (
        dd_speedup_max / dd_speedup_threshold
    ) - 1.0
    dd_incr = dd_speedup_rate / n_x

    v_rx_cum = np.zeros((n_x,)) * np.nan
    v_rx_max = np.zeros((n_x,)) * np.nan
    v_rx_hwm = np.zeros((n_x,)) * np.nan
    v_rx_dd = np.zeros((n_x,)) * np.nan

    v_rx_low = np.zeros((n_x,)) * np.nan

    v_i_max = np.zeros((n_x,), dtype=np.int32)
    v_i_dd = np.zeros((n_x,), dtype=np.int32)
    v_i_low = np.zeros((n_x,), dtype=np.int32)

    v_in_dd = np.zeros((n_x,), dtype=np.int32)
    v_in_rec = np.zeros((n_x,), dtype=np.int32)

    v_since_dd = np.zeros((n_x,), dtype=np.int32)
    rx_since_dd = np.zeros((n_x,), dtype=np.float64)

    max_i_sd_sum: float = 0
    max_i_sd_n: int = 0

    dd_i_sd_sum: float = 0
    dd_i_sd_n: int = 0

    t = 0
    while t < n_t:
        if v_null[t]:
            t += 1
            continue
        x = 0
        while x < n_x:
            if v_is_null[t][x]:
                x += 1
                continue

            rx = v_rx[t][x]

            rx_cum = v_rx_cum[x]
            rx_max = v_rx_max[x]
            rx_hwm = v_rx_hwm[x]

            if np.isnan(rx_cum):
                rx_cum = rx
            else:
                rx_cum += rx

            if np.isnan(rx_max):
                rx_max = rx_cum
                rx_hwm = rx_max
                v_i_max[x] = t
                v_rx_dd[x] = 0.0

                allocs[t][x] = 1.0
            elif rx_cum > rx_max:
                rx_max = rx_cum
                rx_hwm = rx_max

                if v_in_rec[x] == 1:
                    v_in_rec[x] = 0

                v_i_max[x] = t
                v_rx_dd[x] = 0.0
            else:
                rx_max = (decay * rx_max) + (alpha * rx_cum)

            v_rx_cum[x] = rx_cum
            v_rx_max[x] = rx_max
            v_rx_hwm[x] = rx_hwm

            x += 1

        # for x in range(n_x):
        #     if v_is_null[t][x]:
        #         continue
        #     elif v_in_rec[x] == 1:
        #         continue
        #     elif v_in_dd[x] == 1:
        #         rx = v_rx[t][x]
        #         rx_since_dd[x] += rx
        #         continue
        #     elif v_i_max[x] == t:
        #         v_rx_dd[x] = 0.
        #         continue
        #     elif v_in_dd[x] == 0:
        #         rx = v_rx[t][x]
        #         dd = v_rx_dd[x] + (rx * 1)
        #         v_rx_dd[x] = dd

        dd_weight = adjust_weight_start(
            v_i_max,
            v_rx_dd < -dd_speedup_threshold,
            threshold_share,
            n_x,
            span,
            dd_incr,
        )
        # - 1

        # h mean the alloc?
        # h mean the dd threshold?
        # continuous alloc?
        # dd (or lack thereof) in recovery as well?

        x = 0
        while x < n_x:
            if v_is_null[t][x]:
                x += 1
                continue
            elif v_in_rec[x] == 1:
                x += 1
                continue
            elif v_in_dd[x] == 1:
                rx = v_rx[t][x]
                rx_since_dd[x] += rx
                x += 1
                continue
            elif v_i_max[x] == t:
                v_rx_dd[x] = 0.0
                x += 1
                continue

            elif v_in_dd[x] == 0:
                rx = v_rx[t][x]
                dd = v_rx_dd[x] + (rx * dd_weight)

                if dd < -threshold:
                    allocs[t][x] = 0.0
                    labels[t][x] = -0.5

                    v_i_dd[x] = t
                    v_i_low[x] = t
                    v_in_dd[x] = 1

                    rx_since_dd[x] = 0
                    v_since_dd[x] = 0
                    v_rx_low[x] = 0

                    labels[v_i_max[x]][x] = 1.0
                elif (
                    continuous
                    and dd < -dd_speedup_threshold
                ):
                    allocs[t][x] = 1 - (dd / (-threshold))
                elif continuous:
                    allocs[t][x] = 1.0

                v_rx_dd[x] = dd

            x += 1

        rec_weight = adjust_weight_start(
            v_i_low,
            (v_in_dd == 1),
            recovery_share,
            n_x,
            window,
            dd_incr,
        )

        x = 0
        while x < n_x:
            if v_is_null[t][x]:
                x += 1
                continue
            elif v_i_max[x] == t:
                x += 1
                continue
            elif v_in_dd[x] == 0:
                x += 1
                continue
            elif v_in_rec[x] == 1:
                x += 1
                continue

            rx_since = rx_since_dd[x]
            rx_low = v_rx_low[x]
            since_dd = v_since_dd[x]

            if rx_since < rx_low - continuation:
                since_dd = 0
                v_rx_low[x] = rx_since
                v_i_low[x] = t
            else:
                since_dd += rec_weight

            v_since_dd[x] = since_dd

            if since_dd >= window:
                allocs[t][x] = 1.0
                labels[t][x] = 0.5

                v_in_rec[x] = 1
                v_in_dd[x] = 0

                labels[v_i_low[x]][x] = -1.0
            # elif continuous and (
            #     since_dd > window / 2
            # ):
            #     allocs[t][x] = since_dd / window

            x += 1

        if t != n_t - 1:
            t += 1
            continue

        x = 0
        while x < n_x:
            if np.isnan(labels[t][x]):
                labels[t][x] = 0
            x += 1

        t += 1

    if np.isnan(labels[0][x]):
        labels[0][x] = 0

    return allocs, labels, rx_cums


# TODO: drop to 0.5 not zero?
# it's more, how many consecutive drawdowns you hit
# the size of the drawdown determines how long you wait?

# ie. trying to avoid the chop in eg. 05, 12, 18 where we're just top and tailing


# something like, sharp drops more likely to reverse, so stay in / wait less time for bounce
# slower drops more likely to continue, wait more time to see recovery


def dd_regime_df(
    df: pl.DataFrame,
    threshold: float,
    span: int,
    window: int,
    continuation: float,
    share_dd: bool,
    share_recovery: bool,
    continuous: bool,
):
    (nd_rx, nd_null, nd_active, v_null_expr, all_null) = (
        cols_with_null_map(df, drop=["date"])
    )
    allocs, labels, rx = basket_dd_regime(
        nd_rx,
        nd_null,
        all_null,
        threshold=threshold,
        span=int(span),
        window=int(window),
        continuation=continuation,
        threshold_share=share_dd == 1,
        recovery_share=share_recovery == 1,
        continuous=continuous == 1,
    )
    return df.pipe(
        repack_to_df,
        f=dict(
            alloc=lambda e: e.forward_fill().clip(0.0, 1.0)
        ),
        pipe=dict(
            alloc=lambda df: df.pipe(
                null_fill_consecutive,
                len(v_null_expr),
                "alloc",
            )
        ),
        null=dict(alloc=v_null_expr),
        **{
            "label": labels,
            "alloc": allocs,
        },
    )


__all__ = [
    "adjust_weight_start",
    "basket_dd_regime",
    "dd_regime_df",
]
