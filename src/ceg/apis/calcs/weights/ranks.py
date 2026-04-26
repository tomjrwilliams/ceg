from typing import Callable
import polars as pl
import numpy as np
import numba as nb
import math

from ...validation import assert_is

from ...pl import *
from ..utils import *

from .sharpe import *
from .dd import *
from .smooth import *
from .norm import *

# ----------------------------------------


@nb.jit()
def calc_basket_calmar(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_all_null: np.ndarray,
    nd_sample: np.ndarray,
    window: float,
):
    nd_v = np.zeros(nd_rx.shape)  # = metric
    nd_v_null = np.zeros(nd_rx.shape, dtype=np.bool)

    n_t: int = nd_rx.shape[0]
    n_x: int = nd_rx.shape[1]

    x = 0
    while x < n_x:
        nd_null_x = nd_null[:, x]
        if window > 1:
            (nd_v[:, x], nd_v_null[:, x]) = (
                calc_calmar_rolling(
                    nd_rx[:, x],
                    nd_null_x,
                    nd_sample[:, x],
                    int(window) * 4,
                    n_t,
                )
            )
        else:
            # if zero, just becomes expanding?
            if window == 0:
                alpha = 0
            else:
                span = (2 / window) - 1
                span = span * 4
                alpha = 2 / (span + 1)
            (nd_v[:, x], nd_v_null[:, x]) = calc_calmar_ewm(
                nd_rx[:, x],
                nd_null_x,
                nd_sample[:, x],
                alpha,
                n_t,
            )
        x += 1

    return nd_v, nd_v_null


@nb.jit()
def calc_basket_sharpe(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    nd_all_null: np.ndarray,
    nd_sample: np.ndarray,
    window: float,
):
    nd_v = np.zeros(nd_rx.shape)  # = metric
    nd_v_null = np.zeros(nd_rx.shape, dtype=np.bool)

    n_t: int = nd_rx.shape[0]
    n_x: int = nd_rx.shape[1]

    x = 0
    while x < n_x:
        if window == 0 or window > 1:
            v_x, v_null_x = calc_sharpe_rolling(
                nd_rx[:, x],
                nd_null[:, x],
                nd_sample[:, x],
                int(window),
                n_t,
            )
        else:
            v_x, v_null_x = calc_sharpe_ewm(
                nd_rx[:, x],
                nd_null[:, x],
                nd_sample[:, x],
                window,
                n_t,
            )

        nd_v[:, x] = v_x
        nd_v_null[:, x] = v_null_x
        x += 1

    return nd_v, nd_v_null


@nb.jit()
def calc_basket_smooth(
    nd_v: np.ndarray,
    nd_null: np.ndarray,
    smoothing: float,
):
    nd_v_smooth = np.zeros(nd_v.shape)
    nd_v_null = np.zeros(nd_v.shape, dtype=np.bool)

    n_t: int = nd_v.shape[0]
    n_x: int = nd_v.shape[1]

    x = 0
    while x < n_x:
        nd_v_x = nd_v[:, x]
        nd_null_x = nd_null[:, x]

        # TODO: smooth metric method other than mean?

        # maybe encode in metric (eg. sharpe_sharpe etc.)
        # else assume is just smooth=mean

        if smoothing == 0:
            return nd_v, nd_null
        elif smoothing < 1:
            (
                nd_v_smooth[:, x],
                nd_v_null[:, x],
            ) = smooth_ewm(
                nd_v_x,
                nd_null_x,
                smoothing,
                n_t,
            )
        elif smoothing == 1:
            (
                nd_v_smooth[:, x],
                nd_v_null[:, x],
            ) = smooth_cum(
                nd_v_x,
                nd_null_x,
                n_t,
            )
        elif smoothing > 1:
            (
                nd_v_smooth[:, x],
                nd_v_null[:, x],
            ) = smooth_rolling(
                nd_v_x,
                nd_null_x,
                int(smoothing),
                n_t,
            )
        else:
            raise ValueError(smoothing)

        x += 1

    return (
        nd_v_smooth,
        nd_v_null,
    )


@nb.jit()
def combine_weights(n_x: int, n_t: int, *nds: np.ndarray):

    res = np.zeros((n_t, n_x))
    for nd in nds:
        x = 0
        while x < n_x:
            v = 0
            t = 0
            while t < n_t:
                vv = nd[t][x]
                if not np.isnan(vv):
                    v = vv
                res[t][x] += v
                t += 1
            x += 1

    if len(nds) == 1:
        return res

    div = 1 / len(nds)

    t = 0
    while t < n_t:
        x = 0
        while x < n_x:
            res[t][x] *= div
            x += 1
        t += 1

    return res


@nb.jit()
def ffill(nd_v: np.ndarray, n_t: int, n_x: int):
    res = np.zeros(nd_v.shape)
    x = 0
    while x < n_x:
        t = 0
        v = 0
        while t < n_t:
            vv = nd_v[t][x]
            if not np.isnan(vv):
                v = vv
            res[t][x] = v
            t += 1
        x += 1
    return res


@nb.jit()
def calc_net_rx(
    nd_rx: np.ndarray,
    nd_null: np.ndarray,
    n_t: int,
    n_x: int,
):
    res = np.zeros(nd_rx.shape)
    res_null = np.zeros(nd_rx.shape, np.bool)

    t = 0
    while t < n_t:
        acc = 0
        n = 0

        x = 0
        while x < n_x:
            if nd_null[t][x]:
                res_null[t][x] = True
                res[t][x] = np.nan
                x += 1
                continue
            rx = nd_rx[t][x]

            acc += rx
            n += 1

            x += 1

        if n >= 1:
            rx_basket = acc / n
        else:
            rx_basket = 0

        while x < n_x:
            if not nd_null[t][x]:
                rx = nd_rx[t][x]
                res[t][x] = rx - rx_basket
            x += 1

        t += 1

    return res, res_null


def weights_rank_df(
    df: pl.DataFrame,
    metric: str,
    sample: str | int,
    window: float,
    smoothing: float,
    norm: str,
    top: int,  # 0 = all
):

    sample = (
        sample if isinstance(sample, str) else int(sample)
    )

    (
        nd_rx,
        nd_null,
        nd_active,
        v_null_expr,
        nd_all_null,
    ) = cols_with_null_map(df, drop=["date"])

    cols = [k for k in df.schema.keys() if k != "date"]

    df = df.pipe(
        with_sample_masks,
        sample,
        cols,
        with_period="period",
    )
    nd_sample = df.select(
        *[f"{k}_{sample}" for k in cols]
    ).to_numpy()

    assert len(nd_sample.shape) == 2, (
        nd_sample.shape,
        df,
    )
    n_t, n_x = nd_sample.shape

    # TODO: try filter to only those that have had positive sign
    # before anything else

    res = {}
    for met in assert_is(metric, str).split("_"):
        met = met.strip()
        if not len(met):
            continue
        if met.startswith("net"):
            nd_rx_, nd_null_ = calc_net_rx(
                nd_rx,
                nd_null,
                n_t,
                n_x,
            )
            met = met[3:]
        else:
            nd_rx_ = nd_rx
            nd_null_ = nd_null

        if met == "sharpe":
            nd_v, nd_v_null = calc_basket_sharpe(
                nd_rx_,
                nd_null_,
                nd_all_null,
                nd_sample,
                assert_is(window, float),
            )
        elif met == "calmar":
            nd_v, nd_v_null = calc_basket_calmar(
                nd_rx_,
                nd_null_,
                nd_all_null,
                nd_sample,
                assert_is(window, float),
            )
        else:
            raise ValueError(met)

        nd_smooth, nd_smooth_null = calc_basket_smooth(
            nd_v, nd_v_null, assert_is(smoothing, float)
        )
        nd_smooth_fill = ffill(nd_smooth, n_t, n_x)
        nd_smooth_fill = np.where(
            nd_active,
            nd_smooth_fill,
            np.ones(nd_smooth_fill.shape) * -10e6,
        )
        # NOTE: so that null values that havent started yet always get sorted after null values that have
        # TODO: needed, or was that a bug in normal_equal?

        nd_smooth_order = np.argsort(
            -nd_smooth_fill, axis=1
        )

        norm = assert_is(norm, str).replace(
            "pr", "pow_rank_"
        )
        if norm == "eq":
            nd_v_norm = norm_equal(
                nd_smooth_fill,
                nd_smooth_order,
                nd_active,
                n_x,
                n_t,
                top=int(top),
            )
        elif norm == "sm":
            nd_v_norm = norm_softmax(
                nd_smooth_fill,
                nd_smooth_order,
                nd_active,
                n_x,
                n_t,
                top=int(top),
            )
        elif str(norm).startswith("pow_rank"):
            ratio = assert_is(
                2.0
                if norm == "pow_rank_2"
                else 1.5
                if norm == "pow_rank_15"
                else 3
                if norm == "pow_rank_3"
                else None
            )
            nd_v_norm = norm_pow_rank(
                nd_smooth_fill,
                nd_smooth_order,
                nd_active,
                n_x,
                n_t,
                ratio=ratio,
                top=int(top),
            )
        else:
            raise ValueError(norm)

        res[metric] = nd_v
        res[f"{metric}_smooth"] = nd_smooth
        res[f"{metric}_norm"] = nd_v_norm

    # TODO: pre smooth or post smooth?

    res["weight"] = combine_weights(
        n_x,
        n_t,
        *[
            nd
            for k, nd in res.items()
            if k.endswith("_norm")
        ],
    )

    return df.pipe(
        repack_to_df,
        f=None,
        pipe=None,
        null=None,
        # **res,
        weight=res["weight"],
        # NOTE: only save weight
    )


# TODO:

# hrp is doable with just returns
# calc covariance (corr?) matrix
# group closest pair, repeat until tree
# at each tree, rescale to constant vol

# mv (mean variance) also
# use formula for pfolio vol given weights and covar
# constrained to maximise mean rx

# param for how often to recalc cov/corr

# in both cases, can presumably also pass prices
# as a second df (i guess very redundant, many same cols?)
# to also constrain by notional
# but would need to be per unit position (ie. if basket, so not straightforward)

__all__ = [
    "calc_basket_calmar",
    "calc_basket_sharpe",
    "calc_basket_smooth",
    "combine_weights",
    "ffill",
    "calc_net_rx",
    "weights_rank_df",
    # "basket_weights_rank",
]
