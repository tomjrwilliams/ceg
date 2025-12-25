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
st = apis.st.library()

from .....apis import t
from .....apis import perf

from .explorer import *

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
                .forward_fill()
                .pct_change()
                .add(1)
                .cum_prod()
                .over(exp.label_key)
                .pipe(apis.pl.null_mask, pl.col(exp.value_key))
                .alias(exp.value_key)
            )
        elif self.norm == "pct" and self.cum:
            df = df.with_columns(
                pl.col(exp.value_key)
                .fill_null(0)
                .add(1)
                .cum_prod()
                .over(exp.label_key)
                .pipe(apis.pl.null_mask, pl.col(exp.value_key))
                .alias(exp.value_key)
            )
        elif self.cum:
            df = df.with_columns(
                pl.col(exp.value_key)
                .cum_sum()
                .over(exp.label_key)
                .pipe(apis.pl.null_mask, pl.col(exp.value_key))
                .alias(exp.value_key)
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

EXPLORER_COMPONENTS["line"] = DailyLinePlot
EXPLORER_COMPONENTS["line_pct"] = DailyLinePlotPct
EXPLORER_COMPONENTS["line_cum"] = DailyLinePlotCum
EXPLORER_COMPONENTS["line_cum_pct"] = DailyLinePlotCumPct
EXPLORER_COMPONENTS["hist"] = DailyHistogram