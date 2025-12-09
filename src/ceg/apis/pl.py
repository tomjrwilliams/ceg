from typing import (
    TypeVar,
    Type,
    Any,
    overload,
)
import polars as pl

T = TypeVar("T")

def assert_is(v: Any, t: Type[T]) -> T:
    assert isinstance(v, t), (v, t)
    return v

def null_mask(e: pl.Expr, mask: pl.Expr) -> pl.Expr:
    return pl.when(mask.is_null()).then(None).otherwise(e)

def select_item(
    df: pl.DataFrame,
    k: str | pl.Expr,
    t: Type[T],
) -> T:
    e = k if not isinstance(k, str) else pl.col(k)
    v = df.select(e).item()
    assert isinstance(v, t), (v, t)
    return v

@overload
def select_items(
    df: pl.DataFrame,
    ks: tuple[str | pl.Expr],
    t: Type[T],
) -> tuple[T]: ...

@overload
def select_items(
    df: pl.DataFrame,
    ks: tuple[str | pl.Expr, str | pl.Expr],
    t: Type[T],
) -> tuple[T, T]: ...

@overload
def select_items(
    df: pl.DataFrame,
    ks: tuple[
        str | pl.Expr, str | pl.Expr, str | pl.Expr
    ],
    t: Type[T],
) -> tuple[T, T, T]: ...

@overload
def select_items(
    df: pl.DataFrame,
    ks: tuple[str | pl.Expr, ...],
    t: Type[T],
) -> tuple[T, ...]: ...

def select_items(
    df: pl.DataFrame,
    ks: tuple[str | pl.Expr, ...],
    t: Type[T],
) -> tuple[T, ...]:
    es = (
        k if not isinstance(k, str) else pl.col(k)
        for k in ks
    )
    return tuple([
        assert_is(df.select(e).item(), t) for e in es
    ])

def drop_zero(
    df: pl.DataFrame,
    col: str,
    drop_nan: bool = True,
    drop_null: bool = True,
    dp: int | None = 3,
) -> pl.DataFrame:
    expr = pl.col(col)
    if drop_nan:
        expr = expr.fill_nan(0)
    if drop_null:
        expr = expr.fill_null(0)
    if dp is not None:
        expr = expr.round(dp)
    return df.filter(expr != 0)

def join_dfs(
    *_dfs: pl.DataFrame,
    **kwargs,
):
    dfs = list(_dfs)
    df, *dfs = dfs
    for df_r in dfs:
        df = df.join(
            df_r,**kwargs
        )
    return df

__all__ = [
    "assert_is",
    "null_mask",
    "select_item",
    "select_items",
    "drop_zero",
]