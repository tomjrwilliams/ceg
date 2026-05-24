from typing import (
    TypeVar,
    Type,
    Any,
    overload,
    Callable,
)
from datetime import date
import polars as pl
import numpy as np
from .validation import assert_is

T = TypeVar("T")


def unpivot_concatenated_keys(
    df: pl.DataFrame,
    variable_names: list[str],
    value_name: str,
    sep: str = "|",
    index: str | None = None,
):
    unpack = lambda k : (
        pl.col("concat")
        .struct.field(k)
        .alias(k)
    )
    return df.unpivot(
        variable_name="concat",
        value_name=value_name,
        index=index,
    ).with_columns(
        pl.col("concat")
        .str.split(by=pl.lit(sep))
        .list.to_struct(fields= variable_names)
        .alias("concat")
    ).select(value_name, *(
        [] if index is None else [index]
    ), *(
        unpack(variable_name) for variable_name in variable_names
    ))


def filter_dates_between(
    df: pl.DataFrame,
    d0: date | None = None,
    d1: date | None = None,
):
    if d0 is not None and d1 is not None:
        if d1 < d0:
            d1 = d0
        df = df.filter(
            pl.col("date").is_between(d0, d1)
        )
    elif d0 is not None:
        df = df.filter(pl.col("date") >= d0)
    elif d1 is not None:
        df = df.filter(pl.col("date") <= d1)
    else:
        pass
    return df


@overload
def expr(
    e: bool | int | float | str | pl.Expr | None,
    default: None = None,
) -> pl.Expr | None: ...


@overload
def expr(
    e: bool | int | float | str | pl.Expr | None,
    default: bool | int | float,
) -> pl.Expr: ...


def expr(
    e: bool | int | float | str | pl.Expr | None,
    default: bool | int | float | None = None,
) -> pl.Expr | None:
    if e is None:
        assert default is not None, dict(e=e)
        return pl.lit(default)
    return (
        e
        if isinstance(e, pl.Expr)
        else pl.lit(e)
        if isinstance(e, (bool, int, float))
        else pl.col(e)
    )


def is_null(e: pl.Expr, or_inf: bool = True):
    e = e.fill_nan(None)  # TODO: parametrise
    if or_inf:
        return e.is_null().or_(e.is_infinite())
    return e.is_null()


def to_numpy_with_null(s: pl.Series):
    s = s.fill_nan(None).cast(pl.Float64)
    # .replace({
    #     np.inf: None,
    #     -np.inf: None
    # })
    return s.to_numpy(), (
        s.is_null() | s.is_infinite()
    ).cast(pl.Boolean).to_numpy()


def null_mask(e: pl.Expr, mask: pl.Expr) -> pl.Expr:
    return pl.when(is_null(mask)).then(None).otherwise(e)


def cal_mask(e: pl.Series | pl.Expr, cal: str):
    return pl.when(pl.col(cal)).then(e).otherwise(None)


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
    ks: tuple[str | pl.Expr, str | pl.Expr, str | pl.Expr],
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
    return tuple(
        [assert_is(df.select(e).item(), t) for e in es]
    )


def select_min_max(
    df: pl.DataFrame, k: str, t: Type[T]
) -> tuple[T, T]:
    return df.pipe(
        select_items,
        (pl.col(k).min(), pl.col(k).max()),
        t=t,
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
    dfs: list[pl.DataFrame],
    align: bool | None = None,
    on: str | list[str] | None = None,
    **kwargs,
):
    if len(dfs) == 1:
        return dfs[0]

    on = on if isinstance(on, list) else [assert_is(on)]

    if align or (len(dfs) > 2 and align is None):
        align_on = set.intersection(
            *[set(df.schema.keys()) for df in dfs]
        )
        assert set(on) == align_on, dict(
            on=on,
            align_on=align_on,
        )

        df = pl.concat(dfs, how="align_full")

    else:
        df, *dfs = dfs
        for df_r in dfs:
            df = df.join(df_r, on=on, **kwargs)
    return df


def partition_by_exprs(
    df: pl.DataFrame,
    with_alias: bool = False,
    with_keys: bool = True,
    as_list: bool = True,
    **exprs: pl.Expr,
):
    aliases = (
        list(exprs.keys())
        if with_alias
        else [f"{k}_" for k in exprs.keys()]
    )
    df = df.with_columns(
        e.alias(k) for e, k in zip(exprs.values(), aliases)
    )
    if with_keys and as_list:
        return [
            (
                {k: v for k, v in zip(exprs.keys(), keys)},
                (
                    df_partn
                    if with_alias
                    else df_partn.drop(*aliases)
                ),
            )
            for keys, df_partn in df.partition_by(
                *aliases,
                maintain_order=True,
                as_dict=True,
            ).items()
        ]
    raise ValueError(
        (
            df,
            with_alias,
            with_keys,
            as_list,
            list(exprs.keys()),
        )
    )


def partition_by_regime(
    df: pl.DataFrame,
    date: str = "date",
    regime: str = "regime",
    window: str = "window",
    with_alias: bool = False,
    with_keys: bool = True,
    as_list: bool = True,
    truncate_dates: bool = True,
    unique: bool = False,
):
    reg = pl.col(regime)
    reg_ff = reg.forward_fill()
    reg_chg = reg_ff.ne(reg_ff.shift()).fill_null(False)
    reg_ind = reg_chg.cast(pl.Int32).cum_sum()

    win_start = pl.col(f"_{window}_start")
    win_end = pl.col(f"_{window}_end")

    df = (
        df.with_columns(reg_ind.alias(f"_{window}"))
        .with_columns(
            pl.col(date)
            .min()
            .over(f"_{window}")
            .alias(f"_{window}_start"),
            pl.col(date)
            .max()
            .over(f"_{window}")
            .alias(f"_{window}_end"),
        )
        .with_columns(
            pl.concat_str(
                win_start, win_end, separator="-"
            ).alias(window)
            if not truncate_dates
            else pl.concat_str(
                pl.concat_str(
                    (win_start.dt.year() - 2000)
                    .cast(pl.String)
                    .str.pad_start(2, "0"),
                    win_start.dt.month()
                    .cast(pl.String)
                    .str.pad_start(2, "0"),
                ),
                pl.concat_str(
                    (win_end.dt.year() - 2000)
                    .cast(pl.String)
                    .str.pad_start(2, "0"),
                    win_end.dt.month()
                    .cast(pl.String)
                    .str.pad_start(2, "0"),
                ),
                separator="-",
            ).alias(window)
        )
        .drop(
            f"_{window}",
            f"_{window}_start",
            f"_{window}_end",
        )
    )
    return df.pipe(
        partition_by_exprs,
        with_alias=with_alias,
        with_keys=with_keys,
        as_list=as_list,
        window=pl.col("window"),
        regime=pl.col("regime"),
    )


f_round = round


def get_row(
    df: pl.DataFrame,
    i: int,
    *keys: str,
    repr: bool = False,
    round: int | None = None,
):
    r: dict = df.row(i, named=True)
    if not len(keys):
        keys = tuple(r.keys())
    if round:
        r = {
            k: (
                v
                if not isinstance(v, float)
                else f_round(v, round)
            )
            for k, v in r.items()
        }
    if repr:
        for k, v in r.items():
            print(k, v)
        return
    return r


def mask_col(
    col: str, is_valid: Callable[[pl.Expr], pl.Expr]
):
    e = pl.col(col).fill_nan(None)
    is_null = is_valid(e).not_()
    return (
        pl.when(
            # prev value is valid
            is_valid(e.forward_fill().shift(1)),
            is_null,
        )
        .then(0)  # first invalid is null
        .when(is_null)
        .then(None)  # else null
        .otherwise(e)
        .alias(col)
    )


def signed_log(e: pl.Expr):
    e = e.fill_nan(None)
    return (
        pl.when(e.is_null())
        .then(None)
        .when(e > 0)
        .then(e.log())
        .when(e < 0)
        .then(e.mul(-1).log().mul(-1))
        .otherwise(None)
    )


def with_sample_masks(
    df: pl.DataFrame,
    sample: str | float,
    cols: list[str],
    with_period: str | None = None,
):
    if sample == "M":
        df = df.with_columns(
            pl.concat_str(
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
                separator="-",
            ).alias(with_period or "_group")
        )
    elif isinstance(sample, (int, float)):
        sample = int(sample)
        df = df.with_columns(
            pl.int_range(0, len(df))
            .floordiv(sample)
            .alias(with_period or "_group")
        )
    else:
        raise ValueError(df)
    df = df.with_columns(
        *(
            pl.col("date")
            .filter(pl.col(k).is_not_null())
            .last()
            .over(with_period or "_group")
            .eq(pl.col("date"))
            .cast(pl.Boolean)
            .fill_null(False)
            .alias(f"{k}_{sample}")
            for k in cols
        )
    )
    if with_period is None:
        return df.drop("_group")
    return df


__all__ = [
    "expr",
    "to_numpy_with_null",
    "cal_mask",
    "null_mask",
    "select_item",
    "select_items",
    "drop_zero",
    "get_row",
    "mask_col",
    "join_dfs",
    "partition_by_regime",
    "partition_by_exprs",
    "signed_log",
    "is_null",
    "with_sample_masks",
]
