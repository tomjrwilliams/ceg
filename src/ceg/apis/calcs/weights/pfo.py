
from typing import Callable
import polars as pl
import numpy as np
import numba as nb
import math

import itertools

import datetime as dt
from frozendict import frozendict

from ...validation import assert_is

from ...pl import *
from ..utils import *

from .sharpe import *
from .dd import *
from .smooth import *
from .norm import *
from .ranks import *
from .corr import *

# ----------------------------------------



def pfo_weights_df(
    d0: dt.date,
    d1: dt.date,
    rx_dfs: list[pl.DataFrame],
    rx_params: list[frozendict],
    params: frozendict,
):
    
    metrics: (
        str | tuple[str, ...]
    ) = params.get("metrics", ())
    if not isinstance(metrics, tuple):
        metrics = metrics,
    samples: (
        str | int | tuple[str | int, ...]
    ) = params.get("samples", ())
    if not isinstance(samples, tuple):
        samples = samples,
    windows: (
        float | tuple[float, ...]
    ) = params.get("windows", ())
    if not isinstance(windows, tuple):
        windows = windows,
    smoothings: (
        float | tuple[float, ...]
    ) = params.get("smoothings", ())
    if not isinstance(smoothings, tuple):
        smoothings = smoothings,
    norms: (
        str | tuple[str, ...]
    ) = params.get("norms", ())
    if not isinstance(norms, tuple):
        norms = norms,
    tops: (
        int | tuple[int, ...]
    ) = params.get("tops", ())
    if not isinstance(tops, tuple):
        tops = tops,

    rx_dfs = [
        df.select(**(
            {
                "date": "date",
                f"rx_{i}": "rx",
                f"px_{i}": "px",
            }
            if "px" in df.schema
            else {"date": "date", f"rx_{i}":"rx"}
        ))
        for i, df in enumerate(rx_dfs)
    ]

    n = len(rx_dfs)

    cols = [f"rx_{i}" for i in range(n)]
    
    df = (
        join_dfs(rx_dfs, align=True, on="date")
        .sort(by="date")
    )
    
    (
        nd_rx, nd_null, nd_active, v_null_expr, nd_all_null
    ) = cols_with_null_map(
        df, cols=cols,
    )

    res_acc = {}

    nd_corrs = {}
    nd_corrs_vol = {}
    nd_corrs_mu = {}
    nd_corrs_orders = {}
    nd_corrs_places = {}
    nd_corrs_nulls = {}
    nd_corrs_ix = {}
    nd_corrs_iy = {}

    n_t, n_x = nd_rx.shape

    if any((
        "hrp" in norm
        or "mvar" in norm
        for norm in norms
    )):
        corr_speeds = []
        for norm in norms:
            if "hrp" in norm or "mvar" in norm:
                speed = int(norm.split("_")[1])
                alpha = 2 / (speed + 1)
                corr_speeds.append(alpha)
        
        for alpha in corr_speeds:
            (
                nd_corrs[alpha],
                nd_corrs_orders[alpha],
                nd_corrs_places[alpha],
                nd_corrs_nulls[alpha],
                nd_corrs_mu[alpha],
                nd_corrs_vol[alpha],
                nd_corrs_ix[alpha],
                nd_corrs_iy[alpha],
            ) = calc_corr_ewm(
                nd_rx,
                nd_null,
                nd_active,
                alpha,
                n_t,
                n_x,
                # ew smooth rate?
            )

    for sample in samples:
        
        df = df.pipe(
            with_sample_masks,
            sample,
            cols,
            with_period="period"
        )
        nd_sample = df.select(*[
            f"{k}_{sample}" for k in cols
        ]).to_numpy()

        assert len(nd_sample.shape) == 2, (
            nd_sample.shape,
            df,
        )
        n_t, n_x = nd_sample.shape

        for metric, window in itertools.product(
            metrics, windows
        ):
            met = metric

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
                    assert_is(window, float)
                )
            elif met == "calmar":
                nd_v, nd_v_null = calc_basket_calmar(
                    nd_rx_,
                    nd_null_,
                    nd_all_null,
                    nd_sample,
                    assert_is(window, float)
                )
            else:
                raise ValueError(met)
            
            # TODO: if met == equal? rx? 
            # ie. just use the norm not the metric

            for smoothing in smoothings:
                
                nd_smooth, nd_smooth_null = calc_basket_smooth(
                    nd_v,
                    nd_v_null,
                    assert_is(smoothing, float)
                )
                nd_smooth_fill = ffill(nd_smooth, n_t, n_x)
                nd_smooth_fill = np.where(
                    nd_active,
                    nd_smooth_fill,
                    np.ones(nd_smooth_fill.shape) * -10e6
                )
                # NOTE: so that null values that havent started yet always get sorted after null values that have
                # TODO: needed, or was that a bug in normal_equal?

                nd_smooth_order = np.argsort(
                    -nd_smooth_fill, axis=1
                )

                for norm, top in itertools.product(
                    norms, tops
                ):
                    if top > n_x:
                        continue
                    if (
                        "hrp" in norm
                        or "mvar" in norm
                    ):
                        norm_s, speed, norm_sample = norm.split("_")
                        alpha = 2 / (int(speed) + 1)
                        norm_sample = int(norm_sample)
                    else:
                        norm_s = norm
                        alpha = None
                    norm_s = (
                        assert_is(norm_s, str)
                        .replace("pr", "pow_rank_")
                    )
                    if norm_s == "eq":
                        nd_v_norm = norm_equal(
                            nd_smooth_fill,
                            nd_smooth_order,
                            nd_active,
                            n_x,
                            n_t,
                            top=int(top)
                        )
                    elif norm_s == "sm":
                        nd_v_norm = norm_softmax(
                            nd_smooth_fill,
                            nd_smooth_order,
                            nd_active,
                            n_x,
                            n_t,
                            top=int(top)
                        )
                    elif norm_s == "hrp":
                        nd_v_norm = norm_hrp(
                            nd_smooth_fill,
                            nd_smooth_order,
                            nd_active,
                            nd_corrs[alpha],
                            nd_corrs_orders[alpha],
                            nd_corrs_nulls[alpha],
                            nd_corrs_mu[alpha],
                            nd_corrs_vol[alpha],
                            nd_corrs_ix[alpha],
                            nd_corrs_iy[alpha],
                            n_x,
                            n_t,
                            top=int(top),
                            window=int(speed),
                            sample=norm_sample,
                        )

                    # TODO: allow split slow corr
                    # from fast vol

                    # ie. calc two cols in the norm_hrp
                    # one for corr, one vol
                    # only return vol

                    # TODO: drawdown parity?
                    elif norm_s == "mvar":
                        nd_v_norm = norm_mvar(
                            nd_smooth_fill,
                            nd_smooth_order,
                            nd_active,
                            nd_corrs[alpha],
                            nd_corrs_orders[alpha],
                            nd_corrs_nulls[alpha],
                            nd_corrs_mu[alpha],
                            nd_corrs_vol[alpha],
                            nd_corrs_ix[alpha],
                            nd_corrs_iy[alpha],
                            n_x,
                            n_t,
                            top=int(top),
                            sample=norm_sample,
                        )
                    elif str(norm_s).startswith("pow_rank"):
                        ratio = assert_is(
                            2. if norm_s == "pow_rank_2"
                            else 1.5 if norm_s == "pow_rank_15"
                            else 3 if norm_s == "pow_rank_3"
                            else None
                        )
                        nd_v_norm = norm_pow_rank(
                            nd_smooth_fill,
                            nd_smooth_order,
                            nd_active,
                            n_x,
                            n_t,
                            ratio=ratio,
                            top=int(top)
                        )
                    else:
                        raise ValueError(norm_s)
                    
                    nd_weight = combine_weights(
                        n_x,
                        n_t,
                        nd_v_norm,
                    )

                    res_acc[weight_key(
                        metric,
                        sample,
                        window,
                        smoothing,
                        norm,
                        top,
                    )] = nd_weight

    if not len(res_acc):
        raise ValueError(dict(
            metrics=metrics,
            samples=samples,
            windows=windows,
            smoothings=smoothings,
            norms=norms,
            tops=tops,
            params=params,
        ))

    return df.pipe(
        repack_to_df,
        f=None,
        pipe=None,
        null=None,
        **res_acc,
        # key=weight_key,i=rx_index
    )

def weight_key(
    metric: str,
    sample: str | int,
    window: float,
    smoothing: float,
    norm: str,
    top: int, # 0 = all
):
    return (
        f"{metric}-{sample}-{window}-{smoothing}-{norm}-{top}"
    )

__all__ = [
    "pfo_weights_df",
    "weight_key",
]