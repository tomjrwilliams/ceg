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

from ..... import apis
from .....apis import t
from .....apis import perf

from .explorer import *

#  ------------------

# TODO: rename perf_from_rx 
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

        st.dataframe(pl.DataFrame(rs))

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
    
#  ------------------

@dataclass(frozen=True)
class DailyBenchmarking(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    # TODO: downsample params?

    def render(self) -> DailyBenchmarking:
        
        df = self.df
        if not len(df):
            return self
        
        exp = self.explorer
        df = self.render_date_filter(df)

        
        return self
        
#  ------------------

# given partn
# (average, std dev, skew, min / max sensitivity)
# box plot

# across partns, per strategy

# of? sharpe? rx?

#  ------------------

# if period
# maybe a table of color shaded sharpe / rx

# if regime
# per period of regime

# likewise for dd vs non dd periods (bull / bear), though there we'd have to group by the same dd referent in partn

#  ------------------

# if mutation:

# plot change in partition aggregates, against mutation
# eg. sharpe against lag per strategy, etc.
# against degree of (relative?) signal noise
# against rebalance day / randomisation
# degree of position cap enforcement / capping itself (ie. percentages around cap up to uncapped (including below cap))

# also:
# similar summary stats of distribution of stats (sharpe)
# across the mutation

#  ------------------

# TODO: multilevel / hierarchical bayesian estimtaes given the partn / mutn? as a way of playing with pymc

#  ------------------

# POSITION

# mean, sd, skew, etc.

#  ------------------

EXPLORER_COMPONENTS["perf"] = DailyPerformance
EXPLORER_COMPONENTS["bench"] = DailyBenchmarking

__all__ = ["DailyPerformance", "DailyBenchmarking"]