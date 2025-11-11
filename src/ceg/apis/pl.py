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