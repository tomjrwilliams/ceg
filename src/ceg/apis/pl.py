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

def select_min_max(
    df: pl.DataFrame, k: str, t: Type[T]
) -> tuple[T, T]:
    return df.pipe(
        select_items, (pl.col(k).min(), pl.col(k).max()), t=t
    )

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

def partition_by_exprs(
    df: pl.DataFrame,
    with_alias: bool = False,
    with_keys: bool = True,
    as_list: bool = True,
    **exprs: pl.Expr,
):
    aliases = (
        list(exprs.keys()) if with_alias
        else [f"{k}_" for k in exprs.keys()]
    )
    df.with_columns(
        e.alias(k) for e, k in zip(
            exprs.values(), aliases
        )
    )
    if with_keys and as_list:
        return [
            (
                {
                    k: v for k, v in zip(
                        exprs.keys(),
                        keys
                    )
                }, (
                    df_partn
                    if with_alias
                    else df_partn.drop(*aliases)
                )
            )
            for keys, df_partn
            in df.partition_by(
                *aliases,
                maintain_order=True,
                as_dict=True,
            ).items()
        ]
    raise ValueError((df, with_alias, with_keys, as_list, list(exprs.keys())))

__all__ = [
    "assert_is",
    "null_mask",
    "select_item",
    "select_items",
    "drop_zero",
]