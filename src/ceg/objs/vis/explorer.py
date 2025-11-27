from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Callable,
)
import json
from pathlib import Path
import polars as pl
import streamlit as st

from ...engine import *
from ... import vis

from ... import apis
from ...apis.st import cache

from . import apps

from dataclasses import field, replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class AppDomain:
    domains: list[Domain]
    domain_sync: dict[str, bool]

    def obj_domain(self, obj: str):
        for d in self.domains:
            if obj in d.objects:
                if not self.domain_sync.get(d.root.path, False):
                    d = d.sync()
                    self.domain_sync[d.root.path] = True
                return d
        raise ValueError((obj))


@dataclass(frozen=True)
class ExplorerObject:
    graph: int | None
    obj: str
    label: str | None
    group: str | None
    params: str | None

@dataclass(frozen=True)
class ExplorerGraph:
    graph: int | None
    type: str | None
    params: str | None

@dataclass(frozen=True)
class ExplorerPage:
    name: str
    type: str
    domain: str

    objects: list[ExplorerObject]
    graphs: list[ExplorerGraph]

    def with_object(
        self,   
        obj: str | ExplorerObject,
        graph: int | None = None,
        label: str | None = None,
        group: str | None = None,
        params: str | None = None,
    ) -> ExplorerPage:
        kw = dict(
            graph=graph,
            label=label,
            group=group,
            params=params,
        )
        if isinstance(obj, ExplorerObject):
            assert (
                graph is None
                and label is None
                and group is None
                and params is None
            ), kw
        else:
            obj = ExplorerObject(
                graph,
                obj,
                label,
                group,
                params,
            )
        return replace(
            self, objects = self.objects + [obj]
        )
    
    def with_graph(
        self,   
        graph: int | ExplorerGraph | None = None,
        type: str | None = None,
        params: str | None = None,
    ) -> ExplorerPage:
        kw = dict(type=type, params=params)
        if isinstance(graph, ExplorerGraph):
            assert type is None and params is None, kw
        elif isinstance(graph, int):
            graph = ExplorerGraph(graph, type, params)
        else:
            graph = len(self.graphs)
            graph = ExplorerGraph(graph, type, params)
        return replace(
            self, graphs = self.graphs + [graph]
        )

@dataclass(frozen=True)
class ExplorerConfig:
    app: str # of the app
    pages: list[ExplorerPage] = field(
        default_factory=list
    )

    def write(self, root: str = "./__local__/apps"):
        dp = Path(root) / self.app
        for page in self.pages:
            name = page.name
            dp_page = dp / name
            dp_page.mkdir(exist_ok=True, parents=True)
            fp_config = dp_page / "config.json"
            fp_domain = dp_page / "domain.json"
            fp_graphs = dp_page / "graphs.parquet"
            fp_objs = dp_page / "rows.parquet"
            with fp_config.open('w+') as f:
                json.dump({"type": page.type}, f)
            with fp_domain.open('w+') as f:
                json.dump({"value": page.domain}, f)
            df_graphs = pl.DataFrame([
                dict(
                    graph=r.graph,
                    type=r.type,
                    params =r.params,
                )
                for r in page.graphs
            ])
            df_objs = pl.DataFrame([
                dict(
                    graph=r.graph,
                    obj=r.obj,
                    label=r.label,
                    group=r.group,
                    params=r.params,
                )
                for r in page.objects
            ])
            df_graphs.write_parquet(fp_graphs)
            df_objs.write_parquet(fp_objs)

    def with_page(
        self,
        page: str | ExplorerPage,
        type: str | None=None,
        domain: str | None=None,
    ) -> ExplorerConfig:
        params = (page, type, domain)
        if isinstance(page, ExplorerPage):
            assert type is None and domain is None, params
        else:
            assert (
                type is not None 
                and domain is not None
            ), params
            page = ExplorerPage(page, type, domain, [], [])
        return replace(
            self, pages = self.pages + [page]
        )
    
    def with_object(
        self,   
        obj: str | ExplorerObject,
        graph: int | None = None,
        label: str | None = None,
        group: str | None = None,
        params: str | None = None,
    ) -> ExplorerConfig:
        page = self.pages[-1]
        return replace(
            self, pages = self.pages[:-1] + [
                page.with_object(
                    obj=obj,
                    graph=graph,
                    label=label,
                    group=group,
                    params=params,
                )
            ]
        )
    
    def with_graph(
        self,   
        graph: int | ExplorerGraph | None = None,
        type: str | None = None,
        params: str | None = None,
    ) -> ExplorerConfig:
        page = self.pages[-1]
        return replace(
            self, pages = self.pages[:-1] + [
                page.with_graph(
                    graph=graph,
                    type=type,
                    params=params,
                )
            ]
        )

@dataclass(frozen=True)
class Project:
    domains: dict[
        str, Callable[[dict], AppDomain]
    ] = field(default_factory=dict)
    config: ExplorerConfig | None = None
    root: str = "./__local__/apps"

    def reset(self):
        if self.config is not None:
            self.config.write(root=self.root)
        return

    def with_domain(
        self,
        name: str,
        domain: Callable[[dict], AppDomain]
    ) -> Project:
        return replace(self, domains={
            **self.domains,
            **{name: domain}
        })
    
    def with_domains(
        self,
        **domains: Callable[[dict], AppDomain]
    ) -> Project:
        return replace(self, domains={
            **self.domains,
            **domains
        })
    
    def with_config(self, config: ExplorerConfig):
        return replace(self, config=config)

    def with_root(self, root: str):
        return replace(self, root=root)

def prepare_app(
    name: str,
    root_fp: str,
    domain_funcs: dict[
        str, Callable[[dict], AppDomain]
    ]
):
    DOMAINS: dict[str, AppDomain] = {}
    DOMAIN_SYNC = {}

    @cache()
    def obj_to_daily_ts(
        domain: str | None, **kwargs: str
    ) -> pl.DataFrame | None:
        if domain is None or not len(domain):
            return None

        assert "obj" in kwargs, kwargs
        obj = kwargs.pop("obj")

        if obj is None:
            raise ValueError(kwargs)
        
        if domain in DOMAINS:
            dom = DOMAINS[domain]
        else:
            dom = domain_funcs[domain](DOMAIN_SYNC)
            DOMAINS[domain] = dom

        d = dom.obj_domain(obj)
        assert d is not None, obj

        o = d.get(obj, t = None)
        
        val_kv = o.value_key_val(d)
        snap_kv = o.snap_key_val(d)
        v = val_kv.get(
            o.path,
            apis.kv.BODY # type: ignore
            # TODO: get_concat for chunked
        )

        snap = snap_kv.get(o.path)
        assert snap is not None, (domain, obj)
        assert v is not None, (domain, obj)
        assert isinstance(v, ValuePlSeries), (
            domain, obj, v
        )
        d0, d1 = snap.first, snap.last # type: ignore
        df = pl.DataFrame([
            v.ser.alias("value"),
        ]).with_columns(
            pl.lit(o.path).alias("key"),
            pl.date_range(d0, d1).alias("date")
        )
        
        params = kwargs.pop("params")
        # TODO: if params is not None (eg. transform, etc.)

        for k in ["group"]:
            if k in kwargs:
                df = df.with_columns(pl.lit(kwargs[k]).alias(k))
        return df
    
    @dataclass(frozen=True)
    class ExploreDailyObjsFromRows(
        vis.ExploreDailyFromRows
    ):

        @property
        def selection(self) -> vis.StateVariable:
            return vis.StateVariable(
                f"{self.key}_obj_selection"
            )
        
        @property
        def selection_dirty(self) -> vis.StateVariable:
            return vis.StateVariable(
                f"{self.key}_obj_dirty"
            )
        
        @property
        def selection_graph(self) -> vis.StateVariable:
            return vis.StateVariable(
                f"{self.key}_obj_selection_graph"
            )
        
        @property
        def selection_label(self) -> vis.StateVariable:
            return vis.StateVariable(
                f"{self.key}_obj_selection_label"
            )
        
        @property
        def selection_group(self) -> vis.StateVariable:
            return vis.StateVariable(
                f"{self.key}_obj_selection_group"
            )
    
        def gen_domain(self):
            return vis.CachedText(
                f"{self.key}_domain",
                f"{self.cache}/domain.json",
                "domain:",
            )

        def pre_render(self):
            domain = self.gen_domain().render()
            d = domain.data.get()
            if d is None or not len(d):
                return
            
            if d in DOMAINS:
                dom = DOMAINS[d]
            else:
                dom = domain_funcs[d](DOMAIN_SYNC)
                DOMAINS[d] = dom

            objs, types = zip(*set(sum([
                [
                    (o, type(obj).__name__)
                    for o, obj in domain.objects.items()
                ]
                for domain in dom.domains
            ], [])))

            obj_df = pl.DataFrame({
                "type": list(types),
                "object": list(objs)
                # types?
            }).sort(by=["type", "object"])

            def click_add():
                objs_fp = Path(
                    f"{self.cache}/rows.parquet"
                )
                obj_sel = self.selection.get()
                if obj_sel is None:
                    return
                obj_rs = obj_sel.get(
                    "selection", {}
                ).get("rows", [])
                if not len(obj_rs):
                    return
                if not objs_fp.exists():
                    return
                if not len(obj_df):
                    return
                obj_col = obj_df.get_column("object").to_list()
                r_df = pl.DataFrame({
                    "obj": [obj_col[r] for r in obj_rs]
                }).filter(
                    pl.col("obj").is_not_null(),
                    pl.col("obj").str.len_chars() > 0,
                ).select(
                    graph=pl.lit(
                        self.selection_graph.get()
                    ),
                    obj=pl.col("obj"),
                    label=pl.lit(
                        self.selection_label.get()
                    ),
                    group=pl.lit(
                        self.selection_group.get()
                    ),
                    params=pl.lit(None),
                )
                objs_df = pl.concat([
                    pl.read_parquet(objs_fp),
                    r_df,
                ], how="diagonal_relaxed")
                objs_df.write_parquet(objs_fp)

            with st.expander("objects:"):
                self.selection.set(st.dataframe(
                    obj_df,
                    on_select="rerun",
                    selection_mode="multi-row",
                    hide_index=True,
                    # key=self.selection.key,
                ))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                self.selection_graph.set(
                    self.selection_graph.get()
                )
                st.text_input(
                    "graph:",
                    value=self.selection_graph.get(),
                    key=self.selection_graph.key
                )
            with c2:
                self.selection_label.set(
                    self.selection_label.get()
                )
                st.text_input(
                    "label:",
                    value=self.selection_label.get(),
                    key=self.selection_label.key
                )
            with c3:
                self.selection_group.set(
                    self.selection_group.get()
                )
                st.text_input(
                    "group:",
                    value=self.selection_group.get(),
                    key=self.selection_group.key
                )
            with c4:
                url = st.context.url
                if url is not None:
                    path = url.split("/")[-1]
                    apps.hidden_link(
                        id=self.selection_dirty.key,
                        path=path,
                    )
                st.button(
                    "add",
                    on_click=click_add,
                    key=self.selection_dirty.key
                )

        def row_to_df(self, r: dict) -> pl.DataFrame | None:
            domain = self.gen_domain().data.get()
            return obj_to_daily_ts(domain, **r)

        def row_to_alias(self, r: dict) -> str:
            return (
                r["label"] 
                if r["label"] is not None
                else r["obj"]
            )

    app = apps.App(name, root_fp, {})
    return vis.legacy.apps.app(
        [app.list_pages()]
        + list(
            app.get_pages(
                daily_cls=ExploreDailyObjsFromRows,
                # ...
            ).values()
        ),
        init_elements=app.init_page
    )
