from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Type,
    Any,
    Callable,
    Literal,
    overload,
    cast,

)
from pathlib import Path
import datetime as dt
import numpy as np
import polars as pl
import streamlit as st
import plotly.express as px

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ...variables import *
from ..core import *
from ..dates import *
from ..dfs import *
from .....apis import t
from .....apis import perf

@dataclass(frozen=True)
class ExploreDailyComponent(Component):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily
    sync: str | None = None

    def render_date_filter(self, df: pl.DataFrame):
        
        exp = self.explorer

        d = pl.col(exp.date_key)

        d0: dt.date = df.select(d.min()).item()
        d1: dt.date = df.select(d.max()).item()

        date_comp = (
            DateRange(
                self.child_key("daily_dates"),
                d0,
                d1,
                sync=self.sync,
            ).render()
        )

        return df.filter(
            d >= date_comp.date_0.get(),
            d <= date_comp.date_1.get(),
        )

    @classmethod
    def parse_params(cls, s: str | None) -> dict:
        return {}

@dataclass(frozen=True)
class ExploreDaily(Component):
    key: str
    cache: str

    schema: dict[str, Type[pl.DataType]]

    partition: str = "graph"

    date_key: str = "date"
    value_key: str = "value"
    label_key: str = "label"
    group_key: str = "group"
    
    def check_df(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.date_key in df:
            pass
        elif "date" in df:
            df = df.rename({"date": self.date_key})
        else:
            raise ValueError((self, df.schema))
        
        if self.value_key in df:
            pass
        elif "value" in df:
            df = df.rename({"value": self.value_key})
        else:
            raise ValueError((self, df.schema))     
           
        return df

@dataclass(frozen=True)
class DailyLinePlot(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    norm: str | None = None
    cum: bool | None = None

    # TODO: downsample params?

    def render(self) -> DailyLinePlot:
        
        df = self.df
        if not len(df):
            return self
        
        exp = self.explorer
        df = self.render_date_filter(df)

        kws = {}

        if self.norm == "pct" and not self.cum:
            df = df.with_columns(
                pl.col(exp.value_key)
                .pct_change()
                .add(1)
                .cum_prod()
                .over(exp.label_key)
            )
        elif self.norm == "pct" and self.cum:
            df = df.with_columns(
                pl.col(exp.value_key)
                .add(1)
                .cum_prod()
                .over(exp.label_key)
            )
        elif self.cum:
            df = df.with_columns(
                pl.col(exp.value_key)
                .cum_sum()
                .over(exp.label_key)
            )
        elif self.norm is None:
            pass
        else:
            raise ValueError(self)

        fig = px.line(
            df,
            x=exp.date_key,
            y=exp.value_key,
            color=exp.label_key,
            **kws,
        )

        st.plotly_chart(fig)
        
        return self

@dataclass(frozen=True)
class DailyLinePlotPct(DailyLinePlot):
    norm: str | None = "pct"


@dataclass(frozen=True)
class DailyLinePlotCum(DailyLinePlot):
    cum: bool | None = True

@dataclass(frozen=True)
class DailyLinePlotCumPct(
    DailyLinePlotPct, DailyLinePlotCum
):
    cum: bool | None = True
    norm: str | None = "pct"

@dataclass(frozen=True)
class DailyHistogram(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    bins: int = 100
    # TODO: downsample params?

    marginal: str | None = "box"

    def render(self) -> DailyHistogram:
        
        df = self.df
        if not len(df):
            return self
        
        # import plotly.figure_factory as ff

        # # Add histogram data
        # x1 = np.random.randn(200) - 2
        # x2 = np.random.randn(200)
        # x3 = np.random.randn(200) + 2

        # # Group data together
        # hist_data = [x1, x2, x3]

        # group_labels = ["Group 1", "Group 2", "Group 3"]

        # # Create distplot with custom bin_size
        # fig = ff.create_distplot(
        #     hist_data, group_labels, bin_size=[0.1, 0.25, 0.5]
        # )
        # st.plotly_chart(fig)

        exp = self.explorer
        df = self.render_date_filter(df)

        fig = px.histogram(
            df,
            x=exp.value_key,
            color=exp.label_key,
            # y = ... (if 2d plot?)
            nbins=self.bins,
            marginal=self.marginal,
        )

        st.plotly_chart(fig)
        
        return self
    
@dataclass(frozen=True)
class DailyBackTest(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    # TODO: key for which of signal, scale, etc.

    rebal: str = "daily"

    def render(self) -> DailyBackTest:
        
        df = self.df
        if not len(df):
            return self

        exp = self.explorer
        df = self.render_date_filter(df)

        if exp.group_key in df.schema:
            df_groups = {
                g: g_df.pivot(
                    on=exp.label_key,
                    index=exp.date_key,
                    values=exp.value_key,
                    # order?
                )
                for (g,), g_df in df.partition_by(
                    exp.group_key,
                    as_dict=True,
                ).items()
            }
        else:
            df_groups = {
                "rx": df.pivot(
                    on=exp.label_key,
                    index=exp.date_key,
                    values=exp.value_key,
                    # order?
                )
            }

        df_rx = df_groups.pop("rx", None)
        assert df_rx is not None, list(
            df_groups.keys()
        )

        ks = [
            k for k in df_rx.columns if k != exp.date_key
        ]
        bt_dfs = []

        for k in ks:
            k_pos, k_rx = f"{k}_pos", f"{k}_rx"
            bt_df = df_rx.select(
                exp.date_key,
                pl.col(k),
                pl.col(k)
                .is_null()
                .not_()
                .alias("calendar")
            )
            g_kwargs = {}
            for g, g_df in df_groups.items():
                
                # TODO: permutations of sig and scale
                # or just scale only

                # include_sig = only or both

                if k not in g_df.schema:
                    continue
                bt_df = bt_df.join(
                    g_df.select(
                        exp.date_key,
                        pl.col(k).pipe(
                            lag_given_group, g
                        ).alias(g),
                    ),
                    on=exp.date_key,
                    how="left",
                )
                g_kwargs[g] = g

            bt_df = bt_df.pipe(
                perf.with_backtest,
                (k_pos, k_rx),
                cum=False,
                rx=k,
                calendar="calendar",
                **g_kwargs,
                unit=1,
            )
            bt_dfs.append(bt_df)
        return self.calc(df_rx, bt_dfs, ks, exp)
    
    def calc(
        self,
        df_rx: pl.DataFrame,
        df_bts: list[pl.DataFrame],
        ks: list[str],
        exp: ExploreDaily,
    ) -> DailyBackTest: ...

def lag_given_group(e: pl.Expr, g: str) -> pl.Expr:
    if g == "scale":
        e_res = e.shift(1, fill_value=0)
    elif g == "signal":
        e_res = e.shift(1, fill_value=0)
    else:
        return e
    e_res = e_res.forward_fill()
    return pl.when(e.is_null()).then(None).otherwise(e_res)

@dataclass(frozen=True)
class DailyBackTestLineRx(DailyBackTest):

    def calc(
        self,
        df_rx: pl.DataFrame,
        df_bts: list[pl.DataFrame],
        ks: list[str],
        exp: ExploreDaily,
    ):
        df = pl.concat((
            df_rx.select(exp.date_key),
            *([
                df_bt.select(
                    pl.col(f"{k}_rx").alias(k)
                )
                for k, df_bt in zip(ks, df_bts)
            ])
        ), how="horizontal").unpivot(
            index=exp.date_key,
            on=ks,
            variable_name="label",
            value_name="rx",
        )

        fig = px.line(
            df,
            x=exp.date_key,
            y="rx",
            color="label",
        )
        st.plotly_chart(fig)

        return self

@dataclass(frozen=True)
class DailyBackTestTableRx(DailyBackTest):

    include_sig: str = "only"

    @classmethod
    def parse_params(cls, s: str | None) -> dict:
        acc = {}
        if s is None:
            return acc
        for kv in s.split(","):
            kv = kv.strip()
            k, v = kv.split("=")
            acc[k] = v
        return {
            # cum": ...,
            # rebal: ...,

        }

    def calc(
        self,
        df_rx: pl.DataFrame,
        df_bts: list[pl.DataFrame],
        ks: list[str],
        exp: ExploreDaily,
    ):
        res = []

        # NOTE: we use pct=False for hwm
        # if RxCum, then use pct

        for k, bt_df in zip(ks, df_bts):

            # TODO: from the k, bt_df
            # should identify if sig and scale
            # or just scale

            # and return no sig as well if relevant

            # ie. we optionally plot a separate variant with just scale
            # only signal flag (as always scale)
            r = perf.perf_row(
                bt_df,
                k=k,
                sig=True,
            )
            res.append(r)

        st.table(pl.DataFrame(res))

        return self


@dataclass(frozen=True)
class DailyPerformance(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    # TODO: downsample params?

    def render(self) -> DailyPerformance:
        
        df = self.df
        if not len(df):
            return self
        
        exp = self.explorer
        df = self.render_date_filter(df)

        rs = []
        for (label,), label_df in df.partition_by(
            exp.label_key,
            as_dict=True
        ).items():
            if "group" in df.schema:
                label_df = label_df.pivot(
                    on="group",
                    values=exp.value_key,
                )
                r = perf.perf_row(
                    label_df,
                    k=label,
                    sig="Y",
                    pos="pos",
                    rx="rx"
                )
            else:
                r = perf.perf_row(
                    label_df,
                    k=label,
                    sig="Y",
                    rx=exp.value_key,
                )
            rs.append(r)

        st.table(pl.DataFrame(rs))

        if "group" in df.schema:
            df = df.filter(pl.col("group") == "rx")

        df_rx = df.with_columns(
            pl.col(exp.value_key)
            .cum_sum()
            .over(exp.label_key)
        )

        kws = {}
        fig = px.line(
            df_rx,
            x=exp.date_key,
            y=exp.value_key,
            color=exp.label_key,
            **kws,
        )

        st.plotly_chart(fig)
        
        return self
# TODO: scatter blot, similar notion as the above

# so the alias will be used to facet split, the groups to plot x and y

# ideally we can share the same y series (eg. returns) for a whole bunch of x (eg. shifted signal)

# such that we get a grid where x is different signals, y is returns, cells are scatter plots of one vs other

# TODO: i guess later on you even want conditional backtest results given signal thresholds (or a pre-given categorical time series), poss still signal modulated

# TODO: conditional (categorical) line plot highlighting, histograms (ie. share bar locs, color by category)



EXPLORER_COMPONENTS: dict[
    str | None, Type[ExploreDailyComponent]
] = {
    "line": DailyLinePlot,
    "line_pct": DailyLinePlotPct,
    "line_cum": DailyLinePlotCum,
    "line_cum_pct": DailyLinePlotCumPct,
    "hist": DailyHistogram,
    "bt_line_rx": DailyBackTestLineRx,
    "bt_table": DailyBackTestTableRx,
    "perf": DailyPerformance
}

@dataclass(frozen=True)
class ExploreFromRows(Component):
    key: str
    cache: str
    schema: dict[str, Type[pl.DataType]]
    
    label_key: str = "label"

    def row_to_df(self, r: dict) -> pl.DataFrame | None: ...

    def row_to_alias(self, r: dict) -> str: ...

    def check_df(self, df: pl.DataFrame):
        return df

    def get_data(self, df: pl.DataFrame):
        
        r_to_df_alias = lambda r: (
            self.row_to_df(r),
            self.row_to_alias(r),
        )
        
        r_df_aliases = [
            (r_df, alias) for (r_df, alias) in map(
                r_to_df_alias, 
                df.iter_rows(named=True),
            ) if r_df is not None
        ]

        if len(r_df_aliases):
            return pl.concat([
                self.check_df(r_df).with_columns(
                    pl.lit(alias)
                    .alias(self.label_key)
                )
                for r_df, alias in r_df_aliases
            ], how = "diagonal_relaxed")
        
        return None

@dataclass(frozen=True)
class ExploreDailyFromRows(
    ExploreDaily,
    ExploreFromRows,
):
    
    def render(self) -> ExploreDailyFromRows:

        rows = CachedDataFrame(
            self.child_key("data"),
            f"{self.cache}/rows.parquet",
            schema=self.schema,
            widths = {
                "graph": 50,
                "obj": 300,
                "label": 200,
                "group": 100,
                "params": 200,
                # TODO: take this from caller
            }
        ).render()

        if rows.df is None:
            return self

        elif self.partition not in rows.df.schema:
            # TODO: or throw?
            return self

        graphs = CachedDataFrame(
            self.child_key("graphs"),
            f"{self.cache}/graphs.parquet",
            schema={
                self.partition: pl.Int32,
                "type": pl.String,
                "params": pl.String,
                # TODO: take this from caller
            },
            widths = {
                self.partition: 50,
                "type": 200,
                "params": 600,
                # TODO: take this from caller
            }
        ).render()

        if graphs.df is None:
            # or throw?
            return self
        elif self.partition not in graphs.df.schema:
            # TODO: or throw?
            return self

        graph_configs = graphs.df.partition_by(
            self.partition,
            as_dict=True,
        )

        sync = st.toggle(
            label="sync:",
            value=False,
            key=self.child_key("sync")
        )

        for part, part_df in sorted(
            rows.df.filter(
                pl.col(self.partition).is_not_null()
            ).partition_by(
                self.partition,
                as_dict=True,
            ).items(),
            key = lambda part_part_df: part_part_df[0] 
        ):
            
            config_df = graph_configs.get(part)

            if config_df is None or not len(config_df):
                config = None
                graph_type = graph_params = None
            else:
                # TODO: warning if len gt 1
                config = config_df.row(0, named=True)
                config.pop("graph", None)
                graph_type = config.pop("type", None)
                graph_params = config.pop("params", None)

            comp_cls = EXPLORER_COMPONENTS.get(
                graph_type, None
            )

            if comp_cls is None:
                continue

            if part is None:
                part_key = "graph"
            else:
                part_key = f"graph_{part}"

            data = self.get_data(part_df)
            if data is None:
                continue

            if (
                self.group_key in data.schema 
                and not len(data.filter(
                    pl.col(self.group_key).is_not_null()
                ))
            ):
                data = data.drop(self.group_key)
            
            try:
                obj = comp_cls(
                    self.child_key(part_key),
                    t.pl.DataFrame.cast(data),
                    self,
                    sync=self.key if sync else None,
                    **({} if config is None else config),
                    **comp_cls.parse_params(graph_params)
                ).render()
            except:
                raise ValueError(data.schema)
                
        return self