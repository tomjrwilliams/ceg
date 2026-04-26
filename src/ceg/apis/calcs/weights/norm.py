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
def norm_softmax(
    nd_v: np.ndarray,
    nd_order: np.ndarray,
    nd_active: np.ndarray,
    n_x: int,
    n_t: int,
    top: int,
):
    res = np.zeros(nd_v.shape)
    res_null = np.zeros(nd_v.shape, dtype=np.bool)

    if top == 0:
        top = n_x

    t = 0
    while t < n_t:
        n_x_t = 0
        cum_exp = 0

        x = 0
        while x < top:
            x_i = nd_order[t][x]
            if not nd_active[t][x_i] or np.isnan(
                nd_v[t][x_i]
            ):
                x += 1
                continue
            cum_exp += math.exp(nd_v[t][x_i])

            n_x_t += 1
            x += 1

        # if all nan, skip row
        if n_x_t == 0 or cum_exp == 0:
            x = 0
            while x < n_x:
                res[t][x] = np.nan
                res_null[t][x] = True
                x += 1
            t += 1
            continue

        x = 0
        while x < top:
            x_i = nd_order[t][x]
            vv = nd_v[t][x_i]
            if np.isnan(vv) or not nd_active[t][x_i]:
                res[t][x_i] = 0
                # if some nan, set zero (else pass through a weight for if one isnt present?)
                # res_null[t][x_i] = True
                x += 1
                continue

            res[t][x_i] = math.exp(vv) / cum_exp
            x += 1
        while x < n_x:
            x_i = nd_order[t][x]
            res[t][x_i] = 0
            x += 1

        t += 1

    return res


@nb.jit()
def norm_pow_rank(
    nd_v: np.ndarray,
    nd_order: np.ndarray,
    nd_active: np.ndarray,
    n_x: int,
    n_t: int,
    ratio: float = 2.0,
    top: int = 0,
):
    res = np.zeros(nd_v.shape)
    res_null = np.zeros(nd_v.shape, dtype=np.bool)

    if top == 0:
        top = n_x

    t = 0
    while t < n_t:
        n = 0
        w_sum = 0
        w = 1

        # TODO: this has to be just for those active?
        x = 0
        while x < top:
            x_i = nd_order[t][x]
            if not nd_active[t][x_i]:
                res[t][x_i] = np.nan
                res_null[t][x_i] = True
                x += 1
                continue
            w *= 1 / ratio
            w_sum += w
            n += 1
            x += 1

        x = 0
        if n >= 1:
            w_norm = 1 / w_sum
            w = 1
            while x < top:
                x_i = nd_order[t][x]
                if nd_active[t][x_i]:
                    w *= 1 / ratio
                    res[t][x_i] = w * w_norm
                x += 1

        while x < n_x:
            x_i = nd_order[t][x]
            res[t][x_i] = 0
            x += 1

        t += 1

    return res


@nb.jit()
def norm_equal(
    nd_v: np.ndarray,
    nd_order: np.ndarray,
    nd_active: np.ndarray,
    n_x: int,
    n_t: int,
    top: int = 0,
):
    res = np.zeros(nd_v.shape)
    res_null = np.zeros(nd_v.shape, dtype=np.bool)

    if top == 0:
        top = n_x

    t = 0
    while t < n_t:
        n = 0
        x = 0
        while x < top:
            x_i = nd_order[t][x]
            if not nd_active[t][x_i]:
                res[t][x_i] = np.nan
                res_null[t][x_i] = True
                x += 1
                continue
            n += 1
            x += 1

        x = 0
        if n >= 1:
            while x < top:
                x_i = nd_order[t][x]
                if nd_active[t][x_i]:
                    res[t][x_i] = 1 / n
                x += 1

        while x < n_x:
            x_i = nd_order[t][x]
            res[t][x_i] = 0
            x += 1

        t += 1

    return res


@nb.jit()
def norm_mvar(
    nd_v: np.ndarray,
    nd_order: np.ndarray,
    nd_active: np.ndarray,
    nd_corr: np.ndarray,
    nd_corr_order: np.ndarray,
    nd_corr_null: np.ndarray,
    nd_mu: np.ndarray,
    nd_vol: np.ndarray,
    nd_ix: np.ndarray,
    nd_iy: np.ndarray,
    n_x: int,
    n_t: int,
    top: int = 0,
    sample: int = 30,
):
    res = np.zeros(nd_v.shape)
    res_null = np.zeros(nd_v.shape, dtype=np.bool)

    if top == 0:
        top = n_x

    t = 0
    while t < n_t:
        n = 0
        x = 0
        while x < top:
            x_i = nd_order[t][x]
            if not nd_active[t][x_i]:
                res[t][x_i] = np.nan
                res_null[t][x_i] = True
                x += 1
                continue
            n += 1
            x += 1

        x = 0
        if n >= 1:
            while x < top:
                x_i = nd_order[t][x]
                if nd_active[t][x_i]:
                    res[t][x_i] = 1 / n
                x += 1

        while x < n_x:
            x_i = nd_order[t][x]
            res[t][x_i] = 0
            x += 1

        t += 1

    return res


@nb.jit()
def norm_hrp(
    nd_v: np.ndarray,
    nd_order: np.ndarray,
    nd_active: np.ndarray,
    nd_corr: np.ndarray,
    nd_corr_order: np.ndarray,
    nd_corr_null: np.ndarray,
    nd_mu: np.ndarray,
    nd_vol: np.ndarray,
    nd_ix: np.ndarray,
    nd_iy: np.ndarray,
    n_x: int,
    n_t: int,
    top: int,
    window: int,
    sample: int = 30,
):
    n_p = nd_corr.shape[1]
    res = np.zeros(nd_v.shape)
    res_null = np.zeros(nd_v.shape, dtype=np.bool)

    if top == 0:
        top = n_x

    w = np.zeros((n_x,))

    prev_n = np.nan

    n_corr = 0

    r_active = np.zeros((n_x,), dtype=np.bool)

    t = 0
    while t < n_t:
        n = 0
        x = 0
        while x < top:
            x_i = nd_order[t][x]
            if not nd_active[t][x_i]:
                res[t][x_i] = np.nan
                res_null[t][x_i] = True
                x += 1
                continue
            n += 1
            x += 1

        if np.isnan(prev_n):
            prev_n = n

        active_change = False

        x = 0
        if n == 1:
            while x < top:
                x_i = nd_order[t][x]
                # x_i = asset index
                if nd_active[t][x_i]:
                    res[t][x_i] = 1
                x += 1
        elif n > 1:
            x = 0
            while x < n_x:
                x_i = nd_order[t][x]
                if x < top:
                    if not r_active[x_i]:
                        active_change = True
                        r_active[x_i] = True
                else:
                    if r_active[x_i]:
                        r_active[x_i] = False
                        active_change = True
                x += 1
            if (
                t % sample == 0
                or prev_n != n
                or active_change
            ):
                n_corr = 0
                p = 0
                while p < n_p:
                    if not np.isnan(nd_corr[t][p]):
                        n_corr += 1
                    p += 1
                if n_corr > 0:
                    w = hrp_weights(
                        nd_order[t],
                        nd_corr[t],
                        nd_corr_order[t],
                        nd_vol[t],
                        nd_ix,
                        nd_iy,
                        t,
                        top,
                        n,
                    )
            if t < window:
                w = np.ones((n_x,)) / n
            x = 0
            while x < top:
                x_i = nd_order[t][x]
                # x_i = asset index
                wx = w[x_i]
                if nd_active[t][x_i]:
                    if (
                        np.isnan(wx) or wx == 0
                    ) and n_corr > 0:
                        raise ValueError(
                            (
                                w,
                                nd_order[t],
                                nd_corr[t],
                                nd_corr_order[t],
                                nd_vol[t],
                                top,
                                n,
                                t,
                            )
                        )
                    res[t][x_i] = wx
                    # / (
                    #     n
                    #     # * n # TODO: tbc?
                    # )
                x += 1

        prev_n = n

        while x < n_x:
            x_i = nd_order[t][x]
            res[t][x_i] = 0
            x += 1

        t += 1

    return res


# TODO: masked-erms node that takes a signal as well as rx - only calculates the rx where the signal is non zero / null, no decay over the null periods

# TODO: param to hard code correlations to 1 for first pairings

# TODO: pass the signals / mask as an extra argument, when active, weight is assumed zero, we skip the contribution to pfol weight / weigtht rescale for that iteration

# TODO: vol with signal, we set the rx to nan not null so it gets skipped from the vol calc / decay / corr


@nb.jit()
def hrp_weights(
    nd_order: np.ndarray,
    nd_corr: np.ndarray,
    nd_corr_order: np.ndarray,
    nd_vol: np.ndarray,
    nd_ix: np.ndarray,
    nd_iy: np.ndarray,
    t: int,
    top: int,
    n_active: int,
    sample: int = 30,
):
    n_p = nd_ix.shape[0]
    n_x = nd_vol.shape[0]

    nd_included_assets = np.zeros(
        nd_vol.shape, dtype=np.bool
    )
    x = 0
    while x < top:
        i_asset = nd_order[x]
        nd_included_assets[i_asset] = True
        x += 1

    weights = np.zeros(
        (
            n_x,
            n_x,
        )
    )

    nd_pf_map = np.zeros((n_x,), dtype=np.int32) - 1

    # TODO: weights has to be matrix

    last_agg: int = -1
    p: int = 0

    while p < n_p:
        i = nd_corr_order[p]

        x: int = nd_ix[i]
        y: int = nd_iy[i]

        if x == y:
            raise ValueError(
                (
                    nd_corr_order,
                    nd_ix,
                    nd_iy,
                    p,
                    i,
                    x,
                    y,
                )
            )

        if np.isnan(nd_corr[i]):
            p += 1
            continue

        if not (
            nd_included_assets[x] and nd_included_assets[y]
        ):
            p += 1
            continue

        pf_x = nd_pf_map[x]
        pf_y = nd_pf_map[y]

        if pf_x >= 0 and pf_y >= 0 and pf_x != pf_y:
            # if in same, already merged?

            # TODO: assert no current overlap?

            xx = 0
            while xx < n_x:
                wy = weights[pf_y][xx]
                wx = weights[pf_x][xx]

                if wy != 0 and wx != 0:
                    raise ValueError(
                        (weights, pf_x, pf_y, xx, wx, wy)
                    )

                weights[pf_x][xx] = wx + wy
                xx += 1

            pf_vol = agg_vol(
                nd_vol,
                nd_corr,
                weights[pf_x],
                # weights[pf_y],
                nd_ix,
                nd_iy,
                n_active,
            )

            xx = 0
            while xx < n_x:
                # move all of them, not just the new hit
                wx = weights[pf_x][xx]
                if wx != 0:
                    nd_pf_map[xx] = pf_x
                weights[pf_x][xx] = wx / pf_vol
                xx += 1

            last_agg = pf_x

        elif pf_x >= 0 and pf_y >= 0:
            last_agg = pf_x

        elif pf_x >= 0 and pf_y < 0:
            # pf_corr = pf_corr(
            #     nd_vol,
            #     nd_corr,
            #     weights[pf_x]
            #     nd_ix,
            #     nd_iy,
            #     1 # as we rescale after merge
            # )
            if weights[pf_x][y] != 0:
                raise ValueError((weights, pf_x, y))

            weights[pf_x][y] = 1 / n_active
            pf_vol = agg_vol(
                nd_vol,
                nd_corr,
                weights[pf_x],
                nd_ix,
                nd_iy,
                n_active,
            )

            nd_pf_map[y] = pf_x

            xx = 0
            while xx < n_x:
                weights[pf_x][xx] *= 1 / pf_vol
                xx += 1

            last_agg = pf_x

        elif pf_x < 0 and pf_y >= 0:
            if weights[pf_y][x] != 0:
                raise ValueError((weights, pf_y, x))

            # pf_corr = pf_corr(
            #     nd_vol,
            #     nd_corr,
            #     weights[pf_y]
            #     nd_ix,
            #     nd_iy,
            #     1
            # )

            weights[pf_y][x] = 1 / n_active
            pf_vol = agg_vol(
                nd_vol,
                nd_corr,
                weights[pf_y],
                nd_ix,
                nd_iy,
                n_active,
            )

            nd_pf_map[x] = pf_y

            xx = 0
            while xx < n_x:
                weights[pf_y][xx] *= 1 / pf_vol
                xx += 1

            last_agg = pf_y

        elif pf_x < 0 and pf_y < 0:
            if weights[x][x] != 0:
                raise ValueError((weights, x, x))
            if weights[x][y] != 0:
                raise ValueError((weights, x, y))

            weights[x][x] = 1 / n_active
            weights[x][y] = 1 / n_active

            pf_vol = agg_vol(
                nd_vol,
                nd_corr,
                weights[x],
                nd_ix,
                nd_iy,
                n_active,
            )

            weights[x][x] *= 1 / pf_vol
            weights[x][y] *= 1 / pf_vol

            nd_pf_map[x] = x
            nd_pf_map[y] = x

            last_agg = x
        else:
            raise ValueError((pf_x, pf_y))

        p += 1

    res_w = weights[int(last_agg)]

    n_w = 0
    x = 0
    while x < n_x:
        if res_w[x] != 0:
            n_w += 1
        x += 1

    if n_w != n_active:
        raise ValueError(
            (
                weights,
                nd_pf_map,
                last_agg,
                nd_vol,
                nd_corr,
                nd_order,
                nd_corr_order,
                top,
                n_active,
                t,
            )
        )

    return res_w


# TODO: pass asset rx instead of strat where eg. binary signal and sometimes zero
# TODO: pass signal as well, when zero (or close to), ignore from the vol calculation entirely

# ie. aggregate up but dont rescale non vol contributing elements? if signal = 0 for merged asset, skip rescale


@nb.jit()
def agg_pf_vol(
    nd_vol: np.ndarray,
    nd_corr: np.ndarray,
    nd_w_1: np.ndarray,
    nd_w_2: np.ndarray,
    nd_ix: np.ndarray,
    nd_iy: np.ndarray,
    n_active: int,
):
    n_x = nd_vol.shape[0]
    n_p = nd_ix.shape[0]

    # cov_p1_p2 = sum_i(sum_j(w_i_p1 * w_j_p2 * cov_ij))
    # corr_p1_p2 = cov / (vol_p1 * vol_p2)

    # assume p vol == 1 always
    # just sum over sum over w and cov

    acc_var: float = 0

    p = 0
    while p < n_p:
        x = nd_ix[p]
        y = nd_iy[p]

        w_x_1: float = nd_w_1[x]
        w_y_2: float = nd_w_2[y]

        vol_x: float = nd_vol[x]
        vol_y: float = nd_vol[y]

        if np.isnan(vol_x) or np.isnan(vol_y):
            p += 1
            continue

        corr: float = nd_corr[p]

        if vol_x < 0 or vol_y < 0:
            raise ValueError(nd_vol)

        # if corr > 1:
        #     corr = 1
        # elif corr < -1:
        #     corr = -1

        cov = corr * vol_x * vol_y
        acc_var += cov * w_x_1 * w_y_2
        p += 1

    x = 0
    while x < n_x:
        w_x_1: float = nd_w_1[x]
        w_x_2: float = nd_w_2[y]

        vol_x = nd_vol[x]

        if np.isnan(vol_x):
            x += 1
            continue

        acc_var += (vol_x * vol_x) * (w_x_1 * w_x_2)

        x += 1

    if acc_var == 0:
        return 1 / n_active

    acc_vol = math.sqrt(acc_var)
    if acc_vol < 1 / n_active:
        return 1 / n_active
    return acc_vol


@nb.jit()
def agg_vol(
    nd_vol: np.ndarray,
    nd_corr: np.ndarray,
    nd_weights: np.ndarray,
    nd_ix: np.ndarray,
    nd_iy: np.ndarray,
    n_active: int,
):
    n_x = nd_vol.shape[0]
    n_p = nd_ix.shape[0]
    # cov = corr * vol i * vol j
    # sqrt(sum(weight i * weight j * cov ij))
    acc_var: float = 0

    p: int = 0

    while p < n_p:
        x = nd_ix[p]
        y = nd_iy[p]

        w_x: float = nd_weights[x]
        w_y: float = nd_weights[y]

        vol_x: float = nd_vol[x]
        vol_y: float = nd_vol[y]

        corr: float = nd_corr[p]

        if (
            np.isnan(vol_x)
            or np.isnan(vol_y)
            or np.isnan(corr)
        ):
            p += 1
            continue

        if vol_x < 0 or vol_y < 0:
            raise ValueError(nd_vol)

        # if corr > 1:
        #     # value error?
        #     corr = 1
        # elif corr < -1:
        #     corr = -1

        cov = corr * vol_x * vol_y
        acc_var += cov * w_x * w_y
        p += 1

    x = 0
    while x < n_x:
        w_x: float = nd_weights[x]
        vol_x: float = nd_vol[x]
        if np.isnan(vol_x):
            x += 1
            continue
        acc_var += (w_x * w_x) * (vol_x * vol_x)
        x += 1

    if acc_var == 0:
        return 1 / n_active

    acc_vol = math.sqrt(acc_var)
    if acc_vol < 1 / n_active:
        return 1 / n_active
    return acc_vol


@nb.jit()
def pf_corr(
    nd_vol: np.ndarray,
    nd_corr: np.ndarray,
    nd_weights: np.ndarray,
    nd_ix: np.ndarray,
    nd_iy: np.ndarray,
    acc_vol: float,
):
    n_x = nd_vol.shape[0]
    n_p = nd_ix.shape[0]
    # corr_j = sum(weight_i, vol_i, corr_ij) / vol_p
    # = corr between portfolio over i and asset j

    nd_acc_corr = np.zeros(nd_vol.shape)

    x = 0
    while x < n_x:
        if nd_weights[x] == 0:
            acc_corr: float = 0
            p = 0
            while p < n_p:
                px = nd_ix[p]
                py = nd_iy[p]
                w_y = nd_weights[py]
                if px == x and w_y != 0:
                    # corr is with x
                    # y has weight
                    acc_corr += (
                        w_y * nd_vol[py] * nd_corr[p]
                    )
                p += 1
            nd_acc_corr[x] = acc_corr / acc_vol
        x += 1

    return nd_acc_corr


__all__ = [
    "norm_softmax",
    "norm_pow_rank",
    "norm_equal",
    "norm_mvar",
    "norm_hrp",
]
