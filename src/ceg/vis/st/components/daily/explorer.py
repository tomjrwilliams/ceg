from __future__ import annotations
import os
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

from dataclasses import replace, field

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from .....engine import *

from ...variables import *
from ..core import *
from ..dates import *
from ..dfs import *
from ..text import *

from ..... import apis
st = apis.st.library()

from .....apis import assert_is, t, logs, perf

logger, log = logs.logger(
    __file__,
    streams=dict(formatter=True),
)

@dataclass(frozen=True)
class ExploreDaily(Component):
    key: str
    cache: str

    schema: dict[str, Type[pl.DataType]]

    partition: str = "viz"

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
            # TODO: throw only if daily_series
            # raise ValueError((self, df.schema))
            pass
        
        if self.value_key in df:
            pass
        elif "value" in df:
            df = df.rename({"value": self.value_key})
        else:
            # TODO: throw only if daily_series
            # raise ValueError((self, df.schema))
            pass
           
        return df

ROW_CACHE: dict[str, dict[str, pl.DataFrame]] = {}

def row_to_df(
    g: Graph, r: dict
) -> pl.DataFrame | None:
    return obj_to_daily_ts(g, **r)

def row_to_alias(r: dict) -> str:
    return (
        r["label"] 
        if r["label"] is not None
        else r["obj"]
    )

def obj_to_daily_ts(
    g: Graph, **kwargs: str
) -> pl.DataFrame | None:
    # if graph is None or not len(graph):
    #     return None

    # assert "obj" in kwargs, kwargs
    obj = kwargs.pop("obj")

    log("INFO")( loading_obj=obj)

    if obj is None:
        raise ValueError(kwargs)
    
    # # if graph in DOMAINS:
    # g = context.get(Graph, name=graph)
    # else:
    #     dom = graph_funcs[graph](DOMAIN_SYNC)
    #     DOMAINS[graph] = dom

    o = g.get(obj, t = None)
    
    # if hasattr(o, "explorer_df"):
    #     return o.explorer_df(d) # type: ignore
    
    t = kwargs.get("type", "daily_series")

    if isinstance(o, UseRootDataFrameStorage):

        val_kv = o.value_key_val()
        snap_kv = o.snap_key_val()

        snap = snap_kv.get(o.path, Daily.Snap)
        assert snap is not None, (type(o), o)

        v = val_kv.get(
            o.path, ValuePlDataFrame
        )
        assert v is not None, (type(o), o)

        df = v.df

        params = kwargs.pop("params")
        # TODO: if params is not None (eg. transform, etc.)

        for k in ["group"]:
            if k in kwargs:
                df = df.with_columns(pl.lit(kwargs[k]).alias(k))

        return df
    
    elif isinstance(o, Daily):

        val_kv = o.value_key_val()
        snap_kv = o.snap_key_val()

        snap = snap_kv.get(o.path, Daily.Snap)
        assert snap is not None, (type(o), o)

        v = val_kv.get(
            o.path, ValuePlSeries
        )
        assert v is not None, (type(o), o)

        d0, d1 = assert_is(snap.first), assert_is(snap.last)

        try:
            df = v.to_df(
                alias="value",
                pipe=lambda e: e.fill_nan(None)
            )
            df = df.with_columns(
                pl.lit(o.path).alias("key"),
                pl.date_range(d0, d1).alias("date")
            )
        except:
            raise ValueError((
                snap, df
            ))
        
        params = kwargs.pop("params")
        # TODO: if params is not None (eg. transform, etc.)

        for k in ["group"]:
            if k in kwargs:
                df = df.with_columns(pl.lit(kwargs[k]).alias(k))
        return df
    
    raise ValueError((obj, kwargs))

GRAPH_SYNC: dict[str, bool] = {}
PAGE_LOAD: dict[str | None, bool] = {}

def load_page(
    name: str | None,
    Project,
    reset: bool = False
):
    # TODO: root, etc. etc.
    PAGE_LOAD[name] = True
    project = Project.load(target=name)
    if reset:
        project.reset()

@apis.st.cache()
def get_data(
    parent_key: str,
    df_key: str,
    label_key: str,
    graph: str,
):
    with context.graph(graph):
        df = ROW_CACHE[parent_key][df_key]

        if not GRAPH_SYNC.get(graph) or (
            bool(os.environ.get("REGEN"))
        ):
            g = context.get(Graph, name=graph).sync()
            GRAPH_SYNC[graph] = True
        else:
            g = context.get(Graph, name=graph)

        log("INFO")( page=parent_key, df=df_key)
        
        r_to_df_alias = lambda r: (
            row_to_df(g, r),
            row_to_alias(r),
        )
        
        r_df_aliases = [
            (r_df, alias) for (r_df, alias) in map(
                r_to_df_alias, 
                df.iter_rows(named=True),
            ) if r_df is not None
        ]

        if len(r_df_aliases):
            return pl.concat([
                r_df.with_columns(
                    pl.lit(alias)
                    .alias(label_key)
                )
                for r_df, alias in r_df_aliases
            ], how = "diagonal_relaxed")
        
        return None

@dataclass(frozen=True)
class ExploreFromRows(Component):
    key: str
    cache: str
    schema: dict[str, Type[pl.DataType]]
    
    label_key: str = "label"

    dynamic: bool = False

    def row_to_df(self, r: dict) -> pl.DataFrame | None: ...

    def row_to_alias(self, r: dict) -> str: ...

    def check_df(self, df: pl.DataFrame):
        return df

@dataclass(frozen=True)
class ExploreDailyFromRows(
    ExploreDaily,
    ExploreFromRows,
):
    def gen_graph(self):
        return CachedText(
            f"{self.key}_graph",
            f"{self.cache}/graph.json",
            "graph:",
        )

    def get_graph(self):
        
        graph_comp = self.gen_graph()
        if self.dynamic:
            graph = graph_comp.render()
            d = graph.data.get(t=str)
        else:
            d = assert_is(graph_comp.load(set=False))
        return d
    
    def handle_env(self):
        
        args_fp = Path("./src/ceg/app/dev/ARGS.json")
        if args_fp.exists():
            with args_fp.open('r') as f:
                args = json.load(f)
        else:
            args = {}

        logs = args.get("log", [])
        regen = args.get("regen", [])

        for k in logs:
            os.environ[f"LOG.{k}"] = str(True)

        streamlit = bool(os.environ["STREAMLIT"])
        # assert true?

        if len(regen) and streamlit:
            apis.st.library().cache_data.clear()

        for k in regen:
            log("INFO")(regen=k)
            fp = f"./__local__/db/{k}.db"
            
            if Path(fp).exists():
                os.remove(fp)

    def render(self, Project) -> ExploreDailyFromRows:
        log("INFO")(rendering_page=self.key)

        self.handle_env()

        if not PAGE_LOAD.get(self.key):
            load_page(self.key, Project)

        if self.key not in ROW_CACHE:
            ROW_CACHE[self.key] = {}

        rows_comp = CachedDataFrame(
            self.child_key("data"),
            f"{self.cache}/rows.parquet",
            schema=self.schema,
            widths = {
                "viz": 50,
                "obj": 300,
                "label": 200,
                "group": 100,
                "params": 200,
                # TODO: take this from caller
            }
        )
        if self.dynamic:
            rows = rows_comp.render()
            rows_df = rows.df
        else:
            rows_df = rows_comp.merge_schema(rows_comp.load())

        if rows_df is None:
            return self

        elif self.partition not in rows_df.schema:
            # TODO: or throw?
            return self

        vizs_comp = CachedDataFrame(
            self.child_key("vizs"),
            f"{self.cache}/vizs.parquet",
            schema={
                self.partition: pl.Int32,
                "name": pl.String,
                "type": pl.String,
                "params": pl.String,
                # TODO: take this from caller
            },
            widths = {
                self.partition: 50,
                "name": 100,
                "type": 200,
                "params": 600,
                # TODO: take this from caller
            }
        )

        if self.dynamic:
            vizs = vizs_comp.render()
            vizs_df = vizs.df
        else:
            vizs_df = vizs_comp.merge_schema(
                vizs_comp.load()
            )
    
        if vizs_df is None:
            # or throw?
            return self
        elif self.partition not in vizs_df.schema:
            # TODO: or throw?
            return self

        viz_configs = {
            k: v 
            for (k,), v in vizs_df.partition_by(
                self.partition,
                as_dict=True,
            ).items()
        }
        viz_names = {
            k: v.row(0, named=True).get("name", k)
            for k, v in viz_configs.items()
        }
        viz_configs = {
            viz_names.get(k, k): v
            for k, v in viz_configs.items()
        }

        # sync = st.toggle(
        #     label="sync:",
        #     value=False,
        #     key=self.child_key("sync")
        # )

        viz_rows = {
            viz_names.get(
                part, part
            ): partn_df
            for (part,), partn_df 
            in sorted(
                rows_df.filter(
                    pl.col(self.partition).is_not_null()
                ).partition_by(
                    self.partition,
                    as_dict=True,
                ).items(),
                key = lambda part_part_df: part_part_df[0] 
            )
            if part is not None
            and len(assert_is(viz_configs.get(
                viz_names.get(part, part)
            )))
        }

        part = st.segmented_control(
            "viz",
            options=list(viz_rows.keys()),
            selection_mode="single",
            key=f"{self.key}_viz_sel",
            label_visibility="hidden"
        )

        if part is None:
            return self

        # for part, part_df in viz_rows:

            # if part is None:
            #     continue

        part_df = viz_rows[part]

        if part not in ROW_CACHE[self.key]:
            ROW_CACHE[self.key][part] = part_df
                
        config_df = assert_is(viz_configs.get(part))

        # if config_df is None or not len(config_df):
        #     config = None
        #     viz_type = viz_params = None
        # else:

        # TODO: warning if len gt 1
        
        config = config_df.row(0, named=True)
        config.pop("viz", None)
        viz_name = config.pop("name", None)
        viz_type = config.pop("type", None)
        viz_params = config.pop("params", None)

        comp_cls = EXPLORER_COMPONENTS[viz_type]

        # if comp_cls is None:
        #     continue

        # if part is None:
        #     part_key = "viz"
        # else:
        part_key = f"viz_{part}"
    
        data: pl.DataFrame | None = get_data(
            self.key,
            part,
            self.label_key,
            self.get_graph(),
        )

        if data is None:
            return self
    
        data = assert_is(data, pl.DataFrame)

        # TODO: you almost need to ask the viz for conditional filters for data load
        # and then use them to filter the objects before requesting the data
        # so put back the per series cache, remove the viz level 

        if (
            self.group_key in data.schema 
            and not len(data.filter(
                pl.col(self.group_key).is_not_null()
            ))
        ):
            data = data.drop(self.group_key)
        
        log("INFO")( loading_viz=part)

        try:
            obj = comp_cls(
                self.child_key(part_key),
                t.pl.DataFrame.cast(data),
                self,
                # sync=self.key if sync else None,
                sync=None,
                **({} if config is None else config),
                **comp_cls.parse_params(viz_params)
            ).render()
            log("INFO")( ready=part)
        except StopException:
            pass
        except:
            raise ValueError(data.schema)
                
        return self

StopException = (
    st.runtime # type: ignore
    .scriptrunner_utils
    .exceptions
    .StopException
)

@dataclass(frozen=True)
class ExploreDailyComponent(Component):
    key: str
    df: t.pl.DataFrame
    explorer: ExploreDaily
    sync: str | None = None

    def render_date_filter(self, df: pl.DataFrame):
        return df
        
        # exp = self.explorer

        # d = pl.col(exp.date_key)

        # d0: dt.date = df.select(d.min()).item()
        # d1: dt.date = df.select(d.max()).item()

        # date_comp = (
        #     DateRange(
        #         self.child_key("daily_dates"),
        #         d0,
        #         d1,
        #         sync=self.sync,
        #     ).render()
        # )

        # return df.filter(
        #     d >= date_comp.date_0.get(),
        #     d <= date_comp.date_1.get(),
        # )

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
    "load_page",
]