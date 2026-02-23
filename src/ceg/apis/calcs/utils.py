from typing import Callable
import polars as pl
import numpy as np
import numba as nb

from ..pl import is_null, join_dfs
from ..validation import assert_is

@nb.jit()
def calc_hwm(
    v: np.ndarray,
    is_null: np.ndarray,
    alpha: float,
    # TODO: assert float?
):
    i: int = 0
    n: int = len(v)
    res = np.empty(n, dtype=v.dtype)
    acc: float = 0
    alpha_comp = 1 - alpha
    hwm: float = 0.
    while i < n:
        if not is_null[i]:
            vv = v[i]
            if i == 0 or vv > acc:
                acc = vv
                hwm = acc
            else:
                acc = (alpha_comp * acc) + (alpha * vv)
            res[i] = hwm
        i += 1
    return res

# TODO: rename unpack_with_null_map
def cols_with_null_map(
    df: pl.DataFrame,
    drop: list[str] = ["date"],
    cols: list[str] | None = None,
):
    v_null_expr: list[pl.Expr] = []
    v_expr: list[pl.Expr] = []
    v_active_expr: list[pl.Expr] = []
    for k in (
        cols if cols is not None
        else df.drop(*drop).schema
    ):
        v_null_expr.append(
            pl.col(k)
            .pipe(is_null)
            .alias(f"{k}_null")
        )
        v_expr.append(
            pl.when(v_null_expr[-1])
            .then(None)
            .otherwise(pl.col(k).fill_nan(None))
            .alias(k)
        )
        v_active_expr.append(
            v_null_expr[-1]
            .not_()
            .cast(pl.Int32)
            .cum_sum()
            .gt(0)
        )
    all_null = df.with_columns(
        pl.all_horizontal(
            pl.exclude(*drop)
            .fill_nan(None)
            .is_null()
        ).alias("all_null")
    ).get_column("all_null").to_numpy()
    v_null = df.select(
        *v_null_expr
    ).to_numpy()
    v_active = df.select(
        *v_active_expr
    ).to_numpy()
    v_rx = df.select(*v_expr).to_numpy()
    return v_rx, v_null, v_active, v_null_expr, all_null

def null_fill_consecutive(
    df: pl.DataFrame,
    n: int,
    k: str,
):
    v_null = [pl.col(f"null_{i}") for i in range(n)]
    v_expr = [pl.col(f"{k}_{i}") for i in range(n)]
    ks = [f"{k}_{i}" for i in range(n)]
    return df.select(
        "date",
        *[
            pl.when(vn)
            .then(None)
            .when((ve == 0).and_(
                ve.shift() == 0
            ))
            .then(None)
            .otherwise(ve)
            .alias(k)
            for ve, vn, k in zip(
                v_expr,
                v_null,
                ks
            )
        ]
    )

def repack_to_df_(
    df: pl.DataFrame,
    values: np.ndarray,
    prefix: str,
    with_null: list[pl.Expr] | None = None,
    with_cols: list[str] = ["date"],
    f: Callable[[pl.Series], pl.Series] | None = None,
):
    if f is None:
        f = lambda e: e
    if with_null is not None:
        assert len(with_null) == values.shape[1], (
            len(with_null),
            values.shape,
        )
    else:
        with_null = []
    return df.select(
        *with_cols,
        *[
            v_null.alias(f"null_{i}")
            for i, v_null in enumerate(with_null)
        ],
        *[
            f(
                pl.Series(f"{prefix}_{i}", values[:, i])
                .fill_nan(None)
            )            
            .alias(f"{prefix}_{i}")
            for i in range(values.shape[1])
        ],
    )

def repack_to_df(
    df: pl.DataFrame,
    null: (
        list[pl.Expr] | dict[str, list[pl.Expr]] | None
    ) = None,
    f: (
        Callable[[pl.Series], pl.Series]
        | dict[str, Callable[[pl.Series], pl.Series]]
        | None
    ) = None,
    pipe: (
        Callable[..., pl.DataFrame]
        | dict[str, Callable[..., pl.DataFrame]]
        | None
    ) = None,
    **k_vals: np.ndarray | tuple[
        np.ndarray,
        list[pl.Expr] | None,
        Callable[[pl.Series], pl.Series] | None,
        Callable[..., pl.DataFrame] | None,
        dict
    ],
):
    res = []
    for k, vals in k_vals.items():
        if isinstance(vals, tuple):
            (
                vals_nd, null_, f_, f_pipe_, with_kw
            ) = vals
        else:
            vals_nd = vals
            null_ = null
            f_ = f
            f_pipe_ = pipe
            with_kw = {}
        n_x = vals_nd.shape[1]
        val_df = repack_to_df_(
            df,
            vals_nd,
            k,
            with_cols=["date"],
            with_null=(
                null_ if not isinstance(null_, dict)
                else null_.get(k, None)
            ),
            f=(
                f_ if not isinstance(f_, dict)
                else f_.get(k, None)
            ),
        )
        if isinstance(f_pipe_, dict):
            f_pipe_ = f_pipe_.get(k, None)

        if f_pipe_ is not None:
            val_df = f_pipe_(val_df, **with_kw)
        
        res.append(val_df)

    # TODO: align will throw if null cols passed

    return assert_is(
        join_dfs(
            res,
            align=True,
            on="date"
        )
    ).sort(by="date"), [
        (k, i)
        for k in k_vals.keys()
        for i in range(n_x)
    ]

__all__ = [
    "calc_hwm", # TODO: move somehwere else?
    "cols_with_null_map",
    "null_fill_consecutive",
    "repack_to_df",
]