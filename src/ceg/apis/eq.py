import polars as pl
import numpy as np

from .pl import *

def with_splits(
    df: pl.DataFrame,
    raw: str = "raw",
    split: str = "split",
    alias: str = "splits",
):
    e_raw = pl.col(raw)
    e_split = pl.col(split)
    e_raw_pct = pl.col(f"_{raw}_pct")
    e_split_pct = pl.col(f"_{split}_pct")
    df = df.with_columns(
        e_raw
        .fill_nan(None)
        .forward_fill()
        .pct_change()
        .pipe(null_mask, e_raw)
        .alias(f"_{raw}_pct"),
        e_split
        .fill_nan(None)
        .forward_fill()
        .pct_change()
        .pipe(null_mask, e_split)
        .alias(f"_{split}_pct"),
    )
    df = df.with_columns(
        (
            (e_split_pct + 1) / (e_raw_pct + 1)
        ).alias(alias)
    )
    return df.with_columns(
        pl.when(e_raw_pct != e_split_pct)
        .then(pl.col(alias))
        .otherwise(None)
        .alias(alias)
    ).drop(
        f"_{raw}_pct",
        f"_{split}_pct",
    )

# TODO: numba
def calc_div(
    ref: np.ndarray,
    adj: np.ndarray,
    mask: np.ndarray,
):
    res = np.empty(ref.shape, dtype=ref.dtype) * np.nan

    p_ref: float = 0.
    p_adj: float = 0.

    cum_adj: float = 1.

    for t in range(ref.shape[0]):
        if not mask[t]:
            continue

        v_ref: float = ref[t] # type: ignore
        v_adj: float = adj[t] # type: ignore

        if p_ref > 0:
            v_adj = v_adj / cum_adj

            pct_ref = (v_ref / p_ref) - 1
            pct_adj = (v_adj / p_adj) - 1

            diff = np.round(pct_ref - pct_adj, decimals=4)

            if diff != 0:

                div = pct_adj - pct_ref
                res[t] = div

                v_adj = v_adj / (1 - div)
                cum_adj = cum_adj * (1 - div)

        p_ref = v_ref
        p_adj = v_adj

    return res

def with_divs(
    df: pl.DataFrame,
    raw: str = "raw",
    split: str = "split",
    pct: str = "pct",
    alias: str = "divs",
):
    ref = split

    v_ref = df.get_column(ref).to_numpy()
    v_adj = df.get_column(pct).to_numpy()

    v_mask = df.select(
        pl.col(raw).is_not_null().cast(pl.Int32).alias("v")
    ).get_column("v").to_numpy()

    divs = calc_div(v_ref, v_adj, v_mask)

    df = df.with_columns(
        pl.Series(alias, values=divs)
    ).with_columns(
        pl.col(alias).fill_nan(None)
    )

    return df

def with_price(
    df: pl.DataFrame,
    raw: str = "raw",
    split: str = "split",
    pct: str = "pct",
    alias: str = "price",
):
    ratio = "_ratio"

    df = df.pipe(
        with_splits,
        raw=raw,
        split=split,
        alias=ratio
    )
    
    e_raw = pl.col(raw)
    e_split = pl.col(split)
    e_ratio = pl.col(ratio)
    
    df = df.with_columns(
        e_ratio.fill_nan(None)
        .fill_null(1)
        .cum_prod()
        .alias(ratio)
    )
    return df.with_columns(
        (e_raw * e_ratio)
        .pipe(null_mask, e_raw)
        .alias(alias)
    ).drop(ratio)

def with_full(
    df: pl.DataFrame,
    raw: str = "raw",
    split: str = "split",
    pct: str = "pct",
    alias: str = "unsplit",
):
    divs = "_divs"
    price = "_price"
    df = df.pipe(
        with_price,
        raw=raw,
        split=split,
        alias=price
    ).pipe(
        with_divs,
        raw=raw,
        split=split,
        pct=pct,
        alias=divs
    )
    df = df.with_columns(
        pl.col(divs)
        .fill_nan(None)
        .fill_null(0)
        .add(1)
        .cum_prod()
        .alias(divs)
    )
    return df.with_columns(
        (pl.col(price) * pl.col(divs))
        .pipe(null_mask, pl.col(raw))
        .alias(alias)
    ).drop(
        divs, price
    )