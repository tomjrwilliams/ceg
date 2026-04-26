
import polars as pl
import numpy as np

from frozendict import frozendict

from......engine import *

def aggregate_transition_probs(
    df: pl.DataFrame,
    alias: str,
    params: tuple[str, ...] | frozendict[str, str],
    kwargs: frozendict[str, bool | int | float | str],
):
    if isinstance(params, frozendict):
        params = tuple(params.values())

    n = len(params)

    assert n == 2, params

    bwd_k, fwd_k = params

    window_bwd = kwargs.get("window_bwd")
    window_fwd = kwargs.get("window_fwd")

    norm = kwargs.get("norm")

    discretise = kwargs.get("discretise")

    n_bwd = kwargs.get("n_bwd")
    n_fwd = kwargs.get("n_fwd")

    for kw in [
        window_bwd,
        window_fwd,
        n_bwd,
        n_fwd,
        norm,
    ]:
        if kw is None:
            raise ValueError(alias, params, kwargs)
        
    n_bwd = int(n_bwd)
    n_fwd = int(n_fwd)

    days_bwd = window_days(window_bwd)
    days_fwd = window_days(window_fwd)
        
    bwd = pl.col(bwd_k).shift(days_fwd + 1)
    fwd = pl.col(fwd_k)

    # NOTE: align the fwd window shifted state, with the fwd state
    # but where both are backward looking from current state

    df = df.with_columns(*(
        bwd
        .eq(i)
        .alias(f"bwd_{i}")
        for i in range(n_bwd)
    )).with_columns(*(
        fwd
        .eq(i)
        .alias(f"fwd_{i}")
        for i in range(n_fwd)
    ))

    bwd_cols = [f"bwd_{i}" for i in range(n_bwd)]
    fwd_cols = [f"fwd_{i}" for i in range(n_fwd)]

    df = df.with_columns(*(
        pl.col(bwd_c)
        .and_(pl.col(fwd_c))
        .cast(pl.Int32)
        .pipe(
            norm_expr,
            norm=norm,
            n=len(df),
        )
        .alias(f"{bwd_c}-{fwd_c}")
        for bwd_c in bwd_cols
        for fwd_c in fwd_cols
    )).with_columns(*(
        pl.col(bwd_c)
        .cast(pl.Int32)
        .pipe(
            norm_expr,
            norm=norm,
            n=len(df),
        )
        .alias(f"p_{bwd_c}")
        for bwd_c in bwd_cols
    ),*(
        pl.col(fwd_c)
        .cast(pl.Int32)
        .pipe(
            norm_expr,
            norm=norm,
            n=len(df),
        )
        .alias(f"p_{fwd_c}")
        for fwd_c in fwd_cols
    ))

    bwd_fwd_cols = {
        f"{bwd_c}-{fwd_c}": (bwd_c, fwd_c)
        for bwd_c in bwd_cols
        for fwd_c in fwd_cols
    }

    df = df.with_columns(*(
        pl.sum_horizontal(*[
            bwd_fwd for bwd_fwd, (bwd_cc, fwd_cc)
            in bwd_fwd_cols.items()
            if bwd_cc == bwd_c
        ])
        .alias(f"sum_{bwd_c}")
        for bwd_c in bwd_cols
    ), *(
        pl.sum_horizontal(*[
            bwd_fwd for bwd_fwd, (bwd_cc, fwd_cc)
            in bwd_fwd_cols.items()
            if fwd_cc == fwd_c
        ])
        .alias(f"sum_{fwd_c}")
        for fwd_c in fwd_cols
    ))

    df = df.with_columns(*(
        pl.col(bwd_fwd)
        .mul(1 / pl.col(f"sum_{bwd_c}"))
        .alias(f"p_bwd_{bwd_fwd}")
        for bwd_fwd, (bwd_c, fwd_c) in bwd_fwd_cols.items()
        # NOTE: given we start in bwd, where are we likely to end
    ), *(
        pl.col(bwd_fwd)
        .mul(1 / pl.col(f"sum_{fwd_c}"))
        .alias(f"p_fwd_{bwd_fwd}")
        for bwd_fwd, (bwd_c, fwd_c) in bwd_fwd_cols.items()
        # NOTE: given we end in fwd, where are we likely to have started
    ))

    # re add un lagged
    bwd = pl.col(bwd_k)
    fwd = pl.col(fwd_k)
    df = df.with_columns(*(
        bwd
        .eq(i)
        .alias(f"bwd_{i}")
        for i in range(n_bwd)
    )).with_columns(*(
        fwd
        .eq(i)
        .alias(f"fwd_{i}")
        for i in range(n_fwd)
    ))

    return df.select(
        "date",
        pl.concat_list([
            pl.lit(n_bwd),
            pl.lit(n_fwd),
        ] + [
            f"p_{bwd_c}"
            for bwd_c in bwd_cols
        ] + [
            f"p_{fwd_c}"
            for fwd_c in fwd_cols
        ] + [
            bwd_fwd for bwd_fwd in bwd_fwd_cols.keys()
            # NOTE: prob of this intersection
        ] + [
            f"sum_{bwd_c}"
            for bwd_c in bwd_cols
        ] + [
            f"sum_{fwd_c}"
            for fwd_c in fwd_cols
        ] + [
            f"p_bwd_{bwd_fwd}"
            for bwd_fwd in bwd_fwd_cols.keys()
        ] + [
            f"p_fwd_{bwd_fwd}"
            for bwd_fwd in bwd_fwd_cols.keys()
        ] + bwd_cols + fwd_cols).alias(alias)
    )


def get_state_indicator(
    e: pl.Expr,
    n_bwd: int,
    n_fwd: int,
    bwd: int | None = None,
    fwd: int | None = None,
):
    
    bwd_fwd_combs = rank_combinations(n_bwd, n_fwd)
    bwd_fwd_indices = {
        bwd_fwd: i for i, bwd_fwd in enumerate(bwd_fwd_combs)
    }

    n_combs = len(bwd_fwd_indices)

    offset = (
        2 + n_fwd + n_bwd + n_combs + n_bwd + n_fwd + n_combs + n_combs
    )

    if bwd is not None and fwd is not None:
        i = offset + bwd, offset + n_bwd + fwd
    elif bwd is not None:
        i = offset + bwd
    elif fwd is not None:
        i = offset + n_bwd + fwd
    else:
        raise ValueError(n_bwd, n_fwd)
    
    return e.list.get(i)

def get_transition_prob(
    e: pl.Expr,
    n_bwd: int,
    n_fwd: int,
    bwd: int | None = None,
    fwd: int | None = None,
    given: str | None = None,
):

    bwd_fwd_combs = rank_combinations(n_bwd, n_fwd)
    bwd_fwd_indices = {
        bwd_fwd: i for i, bwd_fwd in enumerate(bwd_fwd_combs)
    }

    n_combs = len(bwd_fwd_indices)

    if bwd is None and fwd is None:
        raise ValueError(fwd, fwd, e)
    elif bwd is not None and fwd is None and given is None:
        i = 2 + bwd
    elif fwd is not None and bwd is None and given is None:
        i = 2 + n_fwd + fwd
    elif fwd is not None and bwd is not None and given is None:
        i = 2 + n_fwd + n_bwd + bwd_fwd_indices[(bwd, fwd)]
    else:
        assert given is not None, given
        if bwd is not None and fwd is None and given == "bwd":
            i = 2 + n_fwd + n_bwd + n_combs + bwd
        # elif bwd is not None and fwd is None and given == "fwd":
        # NOTE: if given, cant be none
        #     raise ValueError()
        # elif bwd is None and fwd is not None and given == "bwd":
        # NOTE: if given, cant be none
        #     raise ValueError()
        elif bwd is None and fwd is not None and given == "fwd":
            i = 2 + n_fwd + n_bwd + n_combs + n_bwd + fwd
        elif bwd is not None and fwd is not None and given == "bwd":
            i = 2 + n_fwd + n_bwd + n_combs + n_bwd + n_fwd + (
                bwd_fwd_indices[(bwd, fwd)]
            )
        elif bwd is not None and fwd is not None and given == "fwd":
            i = 2 + n_fwd + n_bwd + n_combs + n_bwd + n_fwd + (
                n_combs
                + bwd_fwd_indices[(bwd, fwd)]
            )
        else:
            raise ValueError(n_bwd, n_fwd, bwd, fwd, given)

    return e.list.get(i)

# TODO: discretise, prob -> weight (including vol scale?)

__all__ = ["aggregate_transition_probs", "get_transition_prob", "get_state_indicator"]