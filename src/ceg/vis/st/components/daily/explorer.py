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

EXPLORER_COMPONENTS: dict[
    str | None, Type[ExploreDailyComponent]
] = {}

__all__ = [
    "ExploreDaily",
    "ExploreFromRows",
    "ExploreDailyFromRows",
    "EXPLORER_COMPONENTS",
    "ExploreDailyComponent",
]