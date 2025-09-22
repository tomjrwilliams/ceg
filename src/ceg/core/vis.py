from __future__ import annotations
from typing import TypeVar
from dataclasses import dataclass, replace
from functools import wraps
import polars as pl
import streamlit as st

import plotly
import plotly.express

T = TypeVar("T")


def not_none(*vs: T | None) -> list[T]:
    return [v for v in vs if v is not None]


def choice_single(*args, f="segmented_control", **kwargs):
    # or f=pills
    return getattr(st, f)(
        *args, **kwargs, selection_mode="single"
    )


def choice_multi(*args, f="segmented_control", **kwargs):
    # or f=pills
    return getattr(st, f)(
        *args, **kwargs, selection_mode="multi"
    )


def is_categorical(dt: pl.DataType):
    return isinstance(dt, (pl.Categorical, pl.Enum))


@dataclass(frozen=True)
class Options:
    col: str | None = None

    @classmethod
    def add(
        cls,
        mp: MultiPivot,
        dim: str,
        col: str,
        col_type: pl.DataType,
    ):
        if col_type == pl.String or is_categorical(col_type):
            # TODO: or boolean?
            mp = OptionsCategorical.add(mp, dim, col)
        elif col_type.is_temporal():
            mp = OptionsTemporal.add(mp, dim, col)
        elif col_type.is_numeric():
            mp = OptionsNumeric.add(mp, dim, col)
        else:
            raise ValueError(
                dict(dim=dim, col=col, col_type=col_type)
            )
        return mp

    def is_categorical(self):
        return False

    def is_temporal(self):
        return False

    def is_numeric(self):
        return False

    def z_expr(self) -> pl.Expr: ...

    def group_expr(self) -> pl.Expr | None:
        return None

    def agg_expr(self) -> pl.Expr | None:
        return None


@dataclass(frozen=True)
class OptionsCategorical(Options):
    col: str | None = None

    @classmethod
    def add(cls, mp: MultiPivot, dim: str, col: str):
        return replace(mp, **{dim: cls(col)})

    def is_categorical(self):
        return True

    def group_expr(self) -> pl.Expr | None:
        assert self.col is not None, self
        return pl.col(self.col)

    def agg_expr(self) -> pl.Expr | None:
        assert self.col is not None, self
        return pl.col(self.col)

    def expr_z(self):
        assert self.col is not None, self
        return pl.col(self.col)


TEMP_FREQ = ["w", "m", "q", "y"]
TEMP_AGG = [
    "first",
    "last",
    # "mid"
    "count",  # eg. if categorical on x (index) on table?
]  # etc.


@dataclass(frozen=True)
class OptionsTemporal(Options):
    col: str | None = None
    transform: str | None = None
    agg: str | None = None
    freq: str | None = None

    @classmethod
    def add(cls, mp: MultiPivot, dim: str, col: str):
        # TODO: eg. transform can be top level date filtering without needing to add a slider
        transform = st.text_input(
            "transform:", key=f"{dim}.transform"
        )
        l, r = st.columns([0.5, 0.5])
        with l:
            freq_choice = choice_single(
                "freq:", TEMP_FREQ, key=f"{dim}.freq"
            )
            agg_choice = choice_single(
                "agg:", TEMP_AGG, key=f"{dim}.agg"
            )
        with r:
            freq_custom = st.text_input(
                "custom:", key=f"{dim}.freq.txt"
            )
            agg_custom = st.text_input(
                "custom:", key=f"{dim}.agg.txt"
            )
        return replace(
            mp,
            **{
                dim: cls(
                    col,
                    transform=transform,
                    freq=freq_choice or freq_custom,
                    agg=agg_choice or agg_custom,
                )
            },
        )

    def is_temporal(self):
        return True

    def group_expr(self):
        if self.freq is None:
            return None
        return self.expr_z().alias(f"{self.col}.freq")

    def agg_expr(self):
        if self.agg is None:
            return None
        assert self.col is not None, self
        if self.agg == "first":
            return pl.col(self.col).first()
        elif self.agg == "last":
            return pl.col(self.col).last()
        else:
            raise ValueError(self)

    def expr_z(self):
        assert self.col is not None, self
        assert self.freq is not None, self
        if self.freq == "w":
            return pl.concat_str(
                pl.col(self.col).dt.year(),
                pl.col(self.col)
                .dt.week()
                .cast(pl.String)
                .str.pad_start(2, "0"),
            )
        elif self.freq == "m":
            return pl.concat_str(
                pl.col(self.col).dt.year(),
                pl.col(self.col).dt.month(),
            )
        elif self.freq == "q":
            return pl.concat_str(
                pl.col(self.col).dt.year(),
                pl.col(self.col).dt.month().floordiv(3),
            )
        elif self.freq == "y":
            return pl.col(self.col).dt.year()
        else:
            raise ValueError(self)


NUM_AGG = ["mean", "stdev"]
NUM_BUCKETS = ["quartile", "decile", "sigmas"]


@dataclass(frozen=True)
class OptionsNumeric(Options):
    col: str | None = None
    transform: str | None = None
    buckets: str | None = None
    agg: str | None = None

    @classmethod
    def add(cls, mp: MultiPivot, dim: str, col: str):
        transform = st.text_input(
            "transform:", key=f"{dim}.transform"
        )
        # arguably pipe syntax is easier to parse than brackets?
        # clip etc. via the transform

        # if we allow multi-select, then need to disambiguate the variables in each transform?
        # and use the transform to define the bi -> uni, so transform is mandatory if more than one

        # so probably do need brackets syntax, with equality for disambiguation
        # but eg. allow | syntax, and optional first equality if only one variable

        # v1 =| ... is equiv v1 = v1 | ...
        # =| ewm_mean(alpha=.3)

        # multiple statements with eg. semicolon
        # if no first equation left referent, assume refers to result, so must be final statement

        # if single variable, assume empty |= refers to v1, else either must be last statement (but has to be then = as no pipe) or have to explicitly specify

        # can then abbreviate the final equals in multi var, eg:
        # v1 =| ewm_mean(v2, alpha=.3); v2 |= ewm_mean(alpha=.2); ewm_corr(v1, v2, alpha=.2)

        l, r = st.columns([0.5, 0.5])
        with l:
            buckets_choice = choice_single(
                "buckets:",
                NUM_BUCKETS,
                key=f"{dim}.buckets",
            )
            agg_choice = choice_single(
                "agg:", NUM_AGG, key=f"{dim}.agg"
            )
        with r:
            buckets_custom = st.text_input(
                "custom:", key=f"{dim}.buckets.txt"
            )
            agg_custom = st.text_input(
                "custom:", key=f"{dim}.agg.txt"
            )
        return replace(
            mp,
            **{
                dim: cls(
                    col,
                    transform=transform,
                    buckets=buckets_choice or buckets_custom,
                    agg=agg_choice or agg_custom,
                )
            },
        )

    def is_numeric(self):
        return True

    def group_expr(self):
        # TODO: buckets
        return None

    def agg_expr(self):
        if self.agg is None:
            return None
        assert self.col is not None, self
        if self.agg == "mean":
            return pl.col(self.col).mean()
        elif self.agg == "stdev":
            return pl.col(self.col).std()
        else:
            raise ValueError(self)

    def expr_z(self):
        assert self.col is not None, self
        assert self.agg is not None, self
        raise ValueError(self)


DF_REPR = ["table", "chart"]


@dataclass(frozen=True)
class MultiPivot:
    df: pl.DataFrame
    x: Options = Options()
    y: Options = Options()
    v: Options = Options()
    z: Options = Options()

    def add(self):
        # TODO: allow multi choice
        # if x / y: all must be treated as categorical (ie. bucketed numerics)
        # effectively just string join the calls assuming tall data?
        # if v: treat as a multi-param function

        df = self.df
        schema = df.schema
        cols = list(schema.keys())

        # TODO: pre create options 1, 2, etc. rows
        # so the shape doesnt change on the page post-select

        cols_x = choice_multi("x:", cols, key="x.cols")
        # on_change=f
        for col in cols_x:
            self = Options.add(self, "x", col, schema[col])

        cols_y = choice_multi("y:", cols, key="y.cols")
        # on_change=f
        for col in cols_y:
            self = Options.add(self, "y", col, schema[col])

        cols_v = choice_multi("v:", cols, key="v.cols")
        # on_change=f
        for col in cols_v:
            self = Options.add(self, "v", col, schema[col])

        # TODO: multi choice, eg. to allow corr, cov, etc.
        # functions given are then conditional on the number selected
        # ie. we need a multi-selection option and a single, with different flow

        # TODO: v2, secondary axis

        cols_z = choice_multi("z:", cols, key="z.cols")
        # on_change=f
        for col in cols_z:
            self = Options.add(self, "z", col, schema[col])

        return self.run()

    def run(self):

        if self.x.col is None:
            return
        if self.v.col is None:
            return

        if self.z.col is not None:
            dfs = self.df.with_columns(
                self.z.z_expr().alias(".z")
            ).partition_by(
                ".z",
                as_dict=True,
                include_key=False,
                maintain_order=True,
            )
        else:
            dfs = {"data": self.df}

        for df_k, df in dfs.items():

            df_repr = choice_single(
                f"{df_k}:",
                DF_REPR,
                key=f"df.{df_k}.repr",
                default=(
                    "chart"
                    if isinstance(self.x, OptionsTemporal)
                    and self.x.freq is None
                    else "table"
                ),
            )

            if df_repr == "chart":
                if self.y.col is None:
                    repr = plotly.express.line(
                        df.select(
                            self.x.col,
                            self.v.col,
                        ),
                        x=self.x.col,
                        y=self.v.col,
                    )
                else:
                    repr = plotly.express.line(
                        df.select(
                            self.x.col,
                            self.v.col,
                            self.y.col,
                        ),
                        x=self.x.col,
                        y=self.v.col,
                        color=self.y.col,  # used for legend
                        # also see symbol and line_group
                        # eg. if many y passed?
                    )
                    repr.update_traces(showlegend=True)
                st.plotly_chart(repr, key=f"df.{df_k}.chart")
            elif df_repr == "table":
                if self.y.col is None:
                    group_exprs = not_none(
                        self.x.group_expr(),
                    )
                    agg_exprs = not_none(
                        self.x.agg_expr(),
                        self.v.agg_expr(),
                    )
                    if len(group_exprs):
                        df = df.group_by(*group_exprs).agg(
                            *agg_exprs
                        )
                    else:
                        df = df.select(self.x.col, self.v.col)
                    st.table(df)
                else:
                    group_exprs = not_none(
                        self.x.group_expr(),
                    )
                    agg_exprs = not_none(
                        self.x.agg_expr(),
                    )
                    y_group = self.y.group_expr()
                    assert y_group is not None, self
                    if len(group_exprs):
                        group_exprs.append(
                            y_group.alias(".cols")
                        )
                        # TODO: the agg needs to go into the pivot?
                        df = (
                            df.group_by(*group_exprs).agg(
                                *agg_exprs
                            )
                        ).pivot(
                            on=".cols",
                            index=self.x.col,
                            values=self.v.col,
                            maintain_order=True,
                        )
                    else:
                        df = (
                            # TODO: if agg required - ie pivot not unique?
                            df.with_columns(
                                y_group.alias(".cols")
                            ).pivot(
                                on=".cols",
                                index=self.x.col,
                                values=self.v.col,
                                maintain_order=True,
                            )
                        )
                    st.table(df)
            else:
                raise ValueError(
                    dict(
                        df=df_k,
                        repr=df_repr,
                    )
                )

        return


# TODO: i think if you have a relational query db over the col keys

# then you can eg. have a ui that grabs all the relation tables
# does the joins for you (?)

# or at least have an add join, give you choice of join tables, andthen which of the selected so far to use for each

# and then query builder for whta to filter on, given the joins


# thats probably the best way to do the generic search interface over keys?

# TODO:

# then given that query, the keys are given the meta data you pick from their col path (?)

# or just the relational query above

# and fed into the vis above, tall
