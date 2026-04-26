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
def corr_init(n_x: int):
    """
    >>> i_x, i_y, order = corr_init(4)
    >>> i_x
    array([0, 0, 0, 1, 1, 2], dtype=uint64)
    >>> i_y
    array([1, 2, 3, 2, 3, 3], dtype=uint64)
    >>> order
    array([0, 1, 2, 3, 4, 5], dtype=uint64)
    """

    n: int = 0
    x: int = 0
    y: int = 0

    while x < n_x:
        n += x
        x += 1

    i_x = np.zeros((n,), dtype=np.uint)
    i_y = np.zeros((n,), dtype=np.uint)
    order = np.zeros((n,), dtype=np.uint)

    x = 0

    i: int = 0
    while x < n_x:
        y = 1 + x
        while y < n_x:
            i_x[i] = x
            i_y[i] = y
            order[i] = i
            i += 1
            y += 1
        x += 1

    return i_x, i_y, order


@nb.jit()
def unpack_diagonal(
    nd_corr: np.ndarray,
    n_t: int,
    n_x: int,
    diag: float = 1.0,
):
    res = np.ones((n_t, n_x, n_x)) * diag

    t = 0
    while t < n_t:
        x = 0
        i = 0
        while x < n_x:
            y = 1 + x
            while y < n_x:
                v = nd_corr[t][i]
                res[t][x][y] = v
                res[t][y][x] = v
                i += 1
                y += 1

            # i += 1
            x += 1
        t += 1

    return res


# @nb.jit()
# def swap_places(
#     i_order: np.ndarray,
#     i_place: np.ndarray,
#     n_i: int,
#     i: int,
#     place: int
# ):

#     # place = n_i - (1 + n_null)

#     place_self = i_place[i]
#     place_i = i_order[place]

#     i_order[place] = i
#     i_order[place_self] = place_i

#     i_place[i] = place
#     i_place[place_i] = place_self

#     return i_order, i_place


@nb.jit()
def calc_corr_ewm(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_active: np.ndarray,
    window: float,
    n_t: int,
    n_x: int,
    warm: int = 180,
    # TODO: clip?
):
    """
    >>> df = pl.DataFrame({
    ...     "v0": [.1, .2, None, .2, .1],
    ...     "v1": [None, None, .1, .2, .1],
    ...     "v2": [.2, .4, None, .4, .2],
    ...     "v3": [-.1, -.2, None, -.2, -.2],
    ... })
    >>> ks = list(df.schema.keys())
    >>> nd_null = df.select((
    ...     pl.col(k).is_null() for k in ks
    ... )).to_numpy()
    >>> nd_active = df.select((
    ...     pl.col(k)
    ...     .is_null()
    ...     .not_()
    ...     .cast(pl.Int32)
    ...     .cum_sum()
    ...     .gt(0)
    ...     for k in ks
    ... )).to_numpy()
    >>> nd_rx = df.to_numpy()
    >>> nd_corr, nd_order, nd_places, nd_null = calc_corr_ewm(
    ...     nd_rx,
    ...     nd_null,
    ...     nd_active,
    ...     0.1,
    ...     len(df),
    ...     len(ks)
    ... )
    >>> unpack = unpack_diagonal
    >>> for corr, order in zip(
    ...     unpack(nd_corr, len(df), len(ks)),
    ...     unpack(nd_places, len(df), len(ks), np.nan)
    ... ):
    ...     print(np.round(np.vstack((corr, order)), 3))
    """
    i_x, i_y, i_order = corr_init(n_x)
    i_place = np.copy(i_order)

    n_i = i_order.shape[0]

    res_mu = np.zeros((n_t, n_x))
    res_vol = np.zeros((n_t, n_x))

    nd_corr = np.zeros(
        (n_t, n_i),
    )
    nd_order = np.zeros((n_t, n_i), dtype=np.uint)
    nd_places = np.zeros((n_t, n_i), dtype=np.uint)
    res_null = np.zeros((n_t, n_i), dtype=np.bool)

    r_mu = np.zeros((n_x,))
    r_var = np.zeros((n_x,)) * np.nan

    r_cov = np.zeros(i_order.shape) * np.nan

    r_corr = np.zeros(i_order.shape) * np.nan
    # r_prod_mu = np.zeros(i_order.shape) * np.nan

    n_first = np.zeros((n_x,)) - 1

    decay = 1 - window
    t: int = 0

    while t < n_t:
        x = 0
        while x < n_x:
            if not nd_active[t][x]:
                res_mu[t][x] = np.nan
                res_vol[t][x] = np.nan
                x += 1
                continue

            if not nd_null[t][x]:
                if n_first[x] < 0:
                    n_first[x] = t

                rx = nd_rx[t][x]

                if np.isnan(r_var[x]):
                    r_mu[x] = rx
                    r_var[x] = rx * rx
                else:
                    r_mu[x] = (decay * r_mu[x]) + (
                        window * rx
                    )
                    r_var[x] = (decay * r_var[x]) + (
                        window * (rx * rx)
                    )

            res_mu[t][x] = r_mu[x]

            x_vol = math.sqrt(r_var[x])

            if t < n_first[x] + warm:
                res_vol[t][x] = np.nan
            elif t == n_first[x] + warm:
                tt = t
                while nd_active[tt][x] and tt >= 0:
                    res_vol[tt][x] = x_vol
                    tt -= 1

            else:
                res_vol[t][x] = x_vol

            x += 1

        x = 0

        i = 0
        n_null = 0

        while x < n_x:
            y = x + 1

            while y < n_x:
                if not (
                    nd_active[t][y] and nd_active[t][x]
                ):
                    res_null[t][i] = True

                    place = n_i - (1 + n_null)

                    i_order, i_place = shuffle_places(
                        i_order,
                        i_place,
                        n_i,
                        i,
                        place,
                        n_null,
                    )

                    # i_places[i] = n_i - (1 + n_null)
                    nd_corr[t][i] = np.nan
                    n_null += 1
                    y += 1
                    i += 1
                    continue

                if x == y:
                    raise ValueError((x, y))

                if not (
                    nd_null[t][x] or nd_null[t][y]
                    # or np.isnan(nd_rx[t][x])
                    # or np.isnan(nd_rx[t][y])
                ):
                    d_x = nd_rx[t][x]
                    #  - r_mu[x]
                    d_y = nd_rx[t][y]
                    #  - r_mu[y]

                    d_prod = d_x * d_y

                    if np.isnan(r_cov[i]):
                        rx_cov = d_prod
                    else:
                        rx_cov = (decay * r_cov[i]) + (
                            window * d_prod
                        )

                    if not np.isnan(rx_cov):
                        r_cov[i] = rx_cov

                    if not (
                        np.isnan(rx_cov)
                        or r_var[x] <= 0
                        or r_var[y] <= 0
                    ):
                        rx_corr = rx_cov / (
                            math.sqrt(r_var[x])
                            * math.sqrt(r_var[y])
                        )

                        nd_corr[t][i] = rx_corr

                        i_order, i_place = insert_place(
                            i_order,
                            i_place,
                            r_corr,
                            rx_corr,
                            n_i,
                            i,
                            n_null,
                        )
                        r_corr[i] = rx_corr

                        y += 1
                        i += 1
                        continue

                # or throw - shouldnt hit?
                res_null[t][i] = True

                # place = n_i - (1 + n_null)

                # i_order, i_place = shuffle_places(
                #     i_order,
                #     i_place,
                #     n_i,
                #     i,
                #     place,
                #     n_null,
                # )

                nd_corr[t][i] = r_corr[i]
                # n_null += 1
                y += 1
                i += 1
                continue

            # i += 1 # sure?
            x += 1

        i = 0
        while i < n_i:
            nd_order[t][i] = i_order[i]
            nd_places[t][i] = i_place[i]
            i += 1

        t += 1

    nd_corr = warm_ffill(nd_corr, nd_order)

    # TODO: separate speed for corr vs vol
    # slow probably better if use same
    # but might be better to use slow corr, fast vol

    return (
        nd_corr,
        nd_order,
        nd_places,
        res_null,
        res_mu * 252,
        res_vol * math.sqrt(252),
        i_x,
        i_y,
    )


@nb.jit()
def warm_ffill(
    vs: np.ndarray, v1: np.ndarray, warm: int = 180
):

    res = np.zeros(vs.shape)
    x = 0

    res_1 = np.zeros(v1.shape, dtype=v1.dtype)

    while x < vs.shape[1]:
        t = 0
        started = -1
        last_v = np.nan

        while t < vs.shape[0]:
            if started == -1:
                last_v1 = v1[t][x]
                if not np.isnan(vs[t][x]):
                    started = t

            if not np.isnan(vs[t][x]):
                if started != -1:
                    last_v1 = v1[t][x]
                last_v = vs[t][x]

            if t == started + warm:
                tt = t
                while tt >= started:
                    res[tt][x] = last_v
                    res_1[tt][x] = last_v1
                    tt -= 1
            else:
                res[t][x] = last_v
                res_1[t][x] = last_v1

            t += 1
        x += 1

    return res


# , res_1


@nb.jit()
def shuffle_places(
    i_order: np.ndarray,
    i_place: np.ndarray,
    n_i: int,
    i: int,
    place: int,
    n_null: int,
):
    place_self = i_place[i]

    if place_self == place:
        res = i_order, i_place
    elif place < place_self:
        res = shuffle_left(
            i_order, i_place, n_i, i, place, n_null
        )
    else:
        res = shuffle_right(
            i_order, i_place, n_i, i, place, n_null
        )
    return res


@nb.jit()
def shuffle_left(
    i_order: np.ndarray,
    i_place: np.ndarray,
    n_i: int,
    i: int,
    place: int,
    n_null: int,
):
    """
    >>> shuffle_left(
    ...     np.array([0, 1, 2]),
    ...     np.array([0, 1, 2]),
    ...     3,
    ...     2,
    ...     0,
    ...     0,
    ... )
    (array([2, 0, 1]), array([1, 2, 0]))
    """

    # place = n_i - (1 + n_null)

    place_self = i_place[i]

    prev_i = int(i)
    p: int = int(place)

    while p < place_self:
        p = int(p)

        place_i: int = int(i_order[p])

        i_order[p] = prev_i
        i_place[int(prev_i)] = p

        prev_i: int = place_i

        p += 1

    i_order[int(p)] = int(prev_i)
    i_place[int(prev_i)] = int(p)

    return i_order, i_place


@nb.jit()
def shuffle_right(
    i_order: np.ndarray,
    i_place: np.ndarray,
    n_i: int,
    i: int,
    place: int,
    n_null: int,
):
    """
    >>> res = shuffle_right(
    ...     np.array([0, 1, 2]),
    ...     np.array([0, 1, 2]),
    ...     3,
    ...     0,
    ...     2,
    ...     0,
    ... )
    >>> res
    >>> shuffle_right(
    ...     *res,
    ...     3,
    ...     0,
    ...     2,
    ...     0,
    ... )
    """

    # place = n_i - (1 + n_null)

    place_self = i_place[i]

    prev_i = int(i)
    p: int = int(place)

    while p > place_self:
        p = int(p)

        place_i: int = int(i_order[p])

        i_order[p] = prev_i
        i_place[int(prev_i)] = p

        prev_i: int = place_i

        p -= 1

    i_order[int(p)] = int(prev_i)
    i_place[int(prev_i)] = int(p)

    return i_order, i_place


@nb.jit()
def insert_place(
    i_order: np.ndarray,
    i_place: np.ndarray,
    row_vs: np.ndarray,
    v: float,
    n_i: int,
    i: int,
    n_null: int,
):
    place = i_place[i]

    if np.isnan(row_vs[i]):
        p = 0
        while p < n_i - n_null:
            p_i = i_order[p]
            p_v = row_vs[p_i]

            if v > p_v or np.isnan(p_v):
                return shuffle_places(
                    i_order,
                    i_place,
                    n_i,
                    i,
                    p,
                    n_null,
                )
            p += 1
        return shuffle_places(
            i_order,
            i_place,
            n_i,
            i,
            p - 1,
            n_null,
        )

    elif v > row_vs[i]:
        p = 0
        while p < place:
            p_i = i_order[p]
            p_v = row_vs[p_i]

            if v > p_v and not np.isnan(p_v):
                return shuffle_places(
                    i_order,
                    i_place,
                    n_i,
                    i,
                    p,
                    n_null,
                )
            p += 1

    elif v < row_vs[i]:
        p = n_i - (1 + n_null)
        while p > place:
            p_i = i_order[p]
            p_v = row_vs[p_i]

            if v < p_v and not np.isnan(p_v):
                return shuffle_places(
                    i_order,
                    i_place,
                    n_i,
                    i,
                    p,
                    n_null,
                )
            p -= 1

    return i_order, i_place


__all__ = ["calc_corr_ewm"]
