
import polars as pl
import numpy as np

from frozendict import frozendict

from......engine import *


def quantize(e: pl.Expr, boundaries: tuple[float, ...]):
    res = pl.when(e.is_null()).then(None)
    i = 0
    for b in boundaries:
        res = res.when(e < b).then(i)
        i += 1
    return res.otherwise(i)


def aggregate_quantize(
    df: pl.DataFrame,
    alias: str,
    params: tuple[str, ...] | frozendict[str, str],
    kwargs: frozendict[str, bool | int | float | str],
):
    if isinstance(params, frozendict):
        params = tuple(params.values())

    n = len(params)

    boundaries = kwargs.get("boundaries", ())
    if len(boundaries) != len(params):
        raise ValueError(alias, params, kwargs)

    method = kwargs.get("method", None)
    diff = kwargs.get("diff", False)
    window = kwargs.get("window", "1D")
    norm = kwargs.get("norm", None)
    reverse = kwargs.get("reverse", None)
    
    n_quant = 1
    for b in boundaries:
        n_quant *= len(b) + 1

    import itertools
    quantize_universe = list(itertools.product(*[
        list(range(len(b) + 1)) for b in boundaries
    ]))
    quantize_map = {
        "||".join(map(str, rs)): i for i, rs in enumerate(quantize_universe)
    }

    if norm:
        df = df.with_columns((
            pl.col(p).pipe(norm_expr, norm, diff = diff, n=len(df)).alias(p)
            for p in params
        ))
        diff = False
    
    if method == "rank":
        df = df.with_columns(
            pl.concat_list(*params)
            .list.eval(pl.element().rank(method='ordinal', descending=True))
            .alias("v")
        ).with_columns((
            pl.col("v").list.get(i).alias(p)
            for i, p in enumerate(params)
        ))
    elif method is None:
        pass
    else:
        raise ValueError(params, kwargs)

    if reverse:
        df = df.reverse()
    try:
        df = df.with_columns((
            (
                pl.col(p)
                if not diff
                else pl.col(p).diff()
            )
            .fill_null(0)
            .rolling_sum(
                window_days(window),
                min_samples=1,
            )
            .alias(p)
            for p in params
        ))
    except:
        raise ValueError(df)
    
    if reverse:
        df = df.reverse()

    df = df.with_columns(*(
        pl.col(p).pipe(quantize, bs).alias(p)
        for p, bs in zip(params, boundaries)
    )).with_columns(
        pl.concat_list(*params)
        .cast(pl.List(pl.Int32))
        .cast(pl.List(pl.String))
        .list.join("||")
        .replace_strict(quantize_map, return_dtype=pl.Int32, default=-1.)
        .alias(alias)
    )
    df_null = df.filter(
        pl.col(alias) == -1.
    )
    if len(df_null):
        raise ValueError(df_null)

    return df

__all__ = ["aggregate_quantize"]