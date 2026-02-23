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

import matplotlib.pyplot as mpl

import plotly.express as px
import plotly.graph_objects as go

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

from .....apis import assert_is, t, perf

from .explorer import *
from .utils import *


@dataclass(frozen=True)
class DailyWatermarks(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    n: int = 2
    width: float = 0.1


    def render(self) -> DailyWatermarks:
        
        df = self.df
        if not len(df):
            return self
        
        exp = self.explorer
        # df = self.render_date_filter(df)

        fig = go.Figure()

        df = df.pipe(
            with_col_split_into,
            "group",
            on="-",
            names=[
                "symbol",
                "span",
                "field",
            ]
        )
        df_state = df.pipe(
            radio_filter_df,
            [
                "symbol",
                "span",
                "field",
            ],
            self.key,
        )

        for (label,), sub_df in df_state.partition_by(
            exp.label_key,
            as_dict=True
        ).items():
            # TODO: manually set colors ourselves?
            for t in px.line(
                sub_df.sort(by="date"),
                x=exp.date_key,
                y=exp.value_key,
                color="group",
            ).update_traces(
                legendgroup=label,
                legendgrouptitle_text=label,
            ).data:
                fig.add_trace(t)

        st.plotly_chart(fig.update_layout(
            legend=dict(groupclick="toggleitem")
        ), key = f"{self.key}_fig_rx")
        
        return self
    
EXPLORER_COMPONENTS["watermarks"] = DailyWatermarks

__all__ = ["DailyWatermarks"]
