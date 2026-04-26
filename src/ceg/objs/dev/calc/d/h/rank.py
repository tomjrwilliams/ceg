
import polars as pl
import numpy as np

from frozendict import frozendict

from......engine import *

# the point with chunkingis that it should make classifying the regime in a clean chunk easier
# eg. by month but it might span multiple regimes
# so, eg. if spy is in a drawdown regime or not is one example

# more generally: a model in which you say, for each new data point, is this a period change (for this asset? or cross sectionally)

# eventually, conditional on current period regime guess

# interesting option is leadership changes in ranking

# eventually aiming for eg. splitting into drawdown regimes (and others), and then classifying based on cum performance vector in period - eg. oil shock, taper tantrum, etc.


def aggregate_rank(
    df: pl.DataFrame,
    alias: str,
    params: tuple[str, ...] | frozendict[str, str],
    kwargs: frozendict[str, bool | int | float | str],
):
    if isinstance(params, frozendict):
        params = tuple(params.values())

    n = len(params)

    method = kwargs.get("method", None)
    diff = kwargs.get("diff", False)
    window = kwargs.get("window", "1D")
    norm = kwargs.get("norm", None)
    reverse = kwargs.get("reverse", None)
    top = kwargs.get("top")

    if not top:
        rank_universe = rank_permutations(n)
        rank_map = {
            "||".join(map(str, [
                r + 1 for r in rs
            ])): i for i, rs in enumerate(rank_universe)
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

    if not top:
        df = df.with_columns(
            pl.concat_list(*params)
            .list.eval(pl.element().rank(method='ordinal', descending=True))
            .cast(pl.List(pl.Int32))
            .cast(pl.List(pl.String))
            .list.join("||")
            .replace_strict(rank_map, return_dtype=pl.Int32, default=-1.)
            .alias(alias)
        )
        df_null = df.filter(
            pl.col(alias) == -1.
        )
        if len(df_null):
            raise ValueError(df_null)
    else:
        df = df.with_columns(
            pl.concat_list(*params)
            .list.eval(pl.element().rank(method='ordinal', descending=True))
            .cast(pl.List(pl.Int32))
            .list.arg_min()
            .alias(alias)
        )

    return df

__all__ = ["aggregate_rank"]