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
class DailyHeatmap(ExploreDailyComponent):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily

    x_key: str = "x"
    y_key: str = "y"
    v_key: str = "v"

    cmap: str = "bwr"

    @classmethod
    def parse_params(cls, s: str | None) -> dict:
        if s is None:
            return {}
        res = {}
        for ss in s.split(","):
            k, v = ss.split("=")
            res[
                {
                    "x": "x_key",
                    "y": "y_key",
                    "v": "v_key",
                }.get(k, k)
            ] = v
        return res

    def render(self) -> DailyHeatmap:

        df = self.df
        if not len(df):
            return self

        exp = self.explorer
        # df = self.render_date_filter(df)

        ks = list(df.schema.keys())

        x = self.x_key
        y = self.y_key
        v = self.v_key

        if x == "label":
            x = exp.label_key
        if y == "label":
            y = exp.label_key
        if v == "value":
            v = exp.value_key

        if ":" in x:
            if x[0] == ":":
                x = ks[: ks.index(x[1:])]
            elif x[-1] == ":":
                x = ks[ks.index(x[:-1]) + 1 :]
            else:
                l, r = x.split(":")
                x = ks[ks.index(l) + 1 : ks.index(r)]
            df = df.with_columns(
                pl.concat_str(
                    *[pl.col(k).cast(pl.String) for k in x],
                    separator="_",
                ).alias("_".join(x))
            )
            x = "_".join(x)

        df = df.with_columns(pl.col(v).round(3))

        v_min, v_max = apis.pl.select_items(
            df.filter(pl.col(v).is_not_null()),
            (pl.col(v).min(), pl.col(v).max()),
            float,
        )

        if v_min < 0 and v_max > 0:
            v_abs_max = max([v_max, -1 * v_min])
            v_min = -v_abs_max
            v_max = v_abs_max
        else:
            v_min = v_max = None

        df_v = df.pivot(
            on=x,  # cols
            index=y,
            values=v,
            # agg = mean?
            sort_columns=True,
            maintain_order=True,
        )

        cmap = mpl.cm.get_cmap(self.cmap)
        st.dataframe(
            df_v.to_pandas()
            .style.background_gradient(
                cmap=cmap, vmin=v_min, vmax=v_max, axis=None
            )
            .format(precision=3)
        )

        return self


EXPLORER_COMPONENTS["heatmap"] = DailyHeatmap

__all__ = ["DailyHeatmap"]
