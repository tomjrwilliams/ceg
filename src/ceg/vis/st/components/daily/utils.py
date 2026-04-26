from typing import Callable
import polars as pl

from ..... import apis

st = apis.st.library()

from ...variables import *


def with_col_split_into(
    df: pl.DataFrame, col: str, on: str, names: list[str]
):
    return (
        df.with_columns(
            pl.col(col)
            .str.split_exact(pl.lit(on), len(names))
            .struct.rename_fields(names)
            .alias(f"_{col}")
        )
        .with_columns(
            *(
                pl.col(f"_{col}")
                .struct.field(name)
                .alias(name)
                for name in names
            )
        )
        .drop(f"_{col}")
    )


def radio_filter_df(
    df: pl.DataFrame,
    cols: list[str],
    parent_key: str,
):

    _, *vals = (
        df.group_by(pl.lit(True))
        .agg(
            *(
                pl.col(name).unique().alias(name)
                for name in cols
            )
        )
        .row(0)
    )

    sels = {}

    for col, opts in zip(cols, vals):
        var = StateVariable(f"{parent_key}_{col}_sel")
        sels[col] = st.segmented_control(
            col,
            opts,
            selection_mode="multi",
            key=var.key,
        )

    return df.filter(
        *(
            pl.all_horizontal(pl.col(col).is_in(opts))
            for col, opts in sels.items()
        )
    )


def quantile_condition_by(
    field_df: pl.DataFrame,
    start: pl.Expr,
    keys: list[str],
    key_name: str,
    quantile: Callable[[pl.Expr, int, str, int], pl.Expr],
    comp: Callable[[pl.Expr, pl.Expr, str], pl.Expr],
):
    cond_df = field_df.with_columns(
        start.cast(pl.Int32).cum_sum().alias(f"_period")
    )
    acc: pl.DataFrame | None = None
    for h_k in keys:
        if h_k not in cond_df.schema.keys():
            continue
        cond_df_ = cond_df.with_columns(
            *(
                pl.col(h_k)
                .fill_nan(None)
                .forward_fill()
                .pipe(
                    comp,
                    pl.col(h_k)
                    .fill_nan(None)
                    .pipe(quantile, i, h_k, len(cond_df))
                    .fill_nan(None)
                    .forward_fill(),
                    h_k,
                )
                .alias(f"{h_k}_{i}")
                for i in range(1, 10)
            )
        )
        cond_df_ = (
            cond_df_.group_by(f"_period")
            .agg(
                pl.col(h_k)
                .fill_nan(None)
                .forward_fill()
                .last()
                .alias(f"v_last"),
                *(
                    pl.col(f"{h_k}_{i}")
                    .any()
                    .replace(
                        True,
                        i,
                        default=None,
                        return_dtype=pl.Int32,
                    )
                    .alias(f"{h_k}_{i}")
                    for i in range(1, 10)
                ),
            )
            .drop("_period")
        )

        cond_df_ = (
            cond_df_.unpivot(
                on=[f"{h_k}_{i}" for i in range(1, 10)],
                index=[f"v_last"],
                value_name="qile",
                variable_name="var",
            )
            .filter(pl.col("qile").is_not_null())
            .with_columns(pl.lit(h_k).alias(key_name))
        )
        if acc is None:
            acc = cond_df_
        else:
            acc = acc.vstack(cond_df_)

    return apis.assert_is(acc)
