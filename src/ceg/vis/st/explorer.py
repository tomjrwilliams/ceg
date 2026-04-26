from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Callable,
    Type,
)
import shutil
import json
from pathlib import Path
import polars as pl

from ... import apis

st = apis.st.library()

from ...engine import *

from ...apis import assert_is
from ...apis.st import cache
from ...apis.logs import logger

from . import apps
from .variables import *
from . import legacy

from dataclasses import field, replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from .components.text import *
from .components.daily.explorer import *

logger, log = logger(
    __file__,
    streams=dict(formatter=True),
)


@dataclass(frozen=True)
class ExplorerObject:
    viz: int | None
    obj: str
    label: str | None
    group: str | None
    params: str | None


@dataclass(frozen=True)
class ExplorerVis:
    viz: int | None
    type: str | None
    params: str | None
    name: str | None


@dataclass(frozen=True)
class ExplorerPage:
    name: str
    type: str
    graph: str

    objects: list[ExplorerObject]
    vizs: list[ExplorerVis]

    def with_object(
        self,
        obj: str | ExplorerObject,
        viz: int | None = None,
        label: str | None = None,
        group: str | None = None,
        params: str | None = None,
    ) -> ExplorerPage:
        kw = dict(
            viz=viz,
            label=label,
            group=group,
            params=params,
        )
        if isinstance(obj, ExplorerObject):
            assert (
                viz is None
                and label is None
                and group is None
                and params is None
            ), kw
        else:
            obj = ExplorerObject(
                viz,
                obj,
                label,
                group,
                params,
            )
        self.objects.append(obj)
        return self
        # return replace(
        #     self, objects = self.objects + [obj]
        # )

    def with_viz(
        self,
        viz: int | ExplorerVis | None = None,
        type: str | None = None,
        params: str | None = None,
        name: str | None = None,
    ) -> ExplorerPage:
        kw = dict(type=type, params=params, name=name)
        if isinstance(viz, ExplorerVis):
            assert type is None and params is None, kw
        elif isinstance(viz, int):
            viz = ExplorerVis(viz, type, params, name)
        else:
            viz = len(self.vizs)
            viz = ExplorerVis(viz, type, params, name)
        self.vizs.append(viz)
        return self
        # return replace(
        #     self, vizs = self.vizs + [viz]
        # )


@dataclass(frozen=True)
class ExplorerConfig:
    app: str  # of the app
    pages: list[ExplorerPage] = field(default_factory=list)

    def write(self, root: str = "./__local__/apps"):
        dp = Path(root) / self.app
        if dp.exists():
            if "__local__" in str(dp):
                assert "frd" not in str(dp), dp
                shutil.rmtree(dp)
                dp.mkdir(exist_ok=True, parents=True)
        for page in self.pages:
            log("INFO")(writing_page=page.name)
            name = page.name
            dp_page = dp / name
            dp_page.mkdir(exist_ok=True, parents=True)
            fp_config = dp_page / "config.json"
            fp_graph = dp_page / "graph.json"
            fp_vizs = dp_page / "vizs.parquet"
            fp_objs = dp_page / "rows.parquet"

            with fp_config.open("w+") as f:
                json.dump({"type": page.type}, f)
            with fp_graph.open("w+") as f:
                json.dump({"value": page.graph}, f)

            def viz_row(r: ExplorerVis):
                return dict(
                    viz=r.viz,
                    name=r.name,
                    type=r.type,
                    params=r.params,
                )

            def obj_row(r: ExplorerObject):
                return dict(
                    viz=r.viz,
                    obj=r.obj,
                    label=r.label,
                    group=r.group,
                    params=r.params,
                )

            df_vizs = pl.DataFrame(map(viz_row, page.vizs))
            df_vizs.write_parquet(fp_vizs)
            objs = list(map(obj_row, page.objects))
            df_objs = pl.DataFrame(objs)
            df_objs.write_parquet(
                fp_objs, compression="lz4"
            )

    def with_page(
        self,
        page: str | ExplorerPage,
        type: str | None = None,
        graph: str | None = None,
    ) -> ExplorerConfig:
        params = (page, type, graph)
        if isinstance(page, ExplorerPage):
            assert type is None and graph is None, params
        else:
            assert type is not None and graph is not None, (
                params
            )
            page = ExplorerPage(page, type, graph, [], [])
        # return replace(
        #     self, pages = self.pages + [page]
        # )
        self.pages.append(page)
        return self

    def with_object(
        self,
        obj: str | ExplorerObject,
        viz: int | None = None,
        label: str | None = None,
        group: str | None = None,
        params: str | None = None,
    ) -> ExplorerConfig:
        page = self.pages[-1]
        page = page.with_object(
            obj=obj,
            viz=viz,
            label=label,
            group=group,
            params=params,
        )
        # return replace(
        #     self, pages = self.pages[:-1] + [page]
        # )
        self.pages[-1] = page
        return self

    def with_viz(
        self,
        viz: int | ExplorerVis | None = None,
        type: str | None = None,
        params: str | None = None,
        name: str | None = None,
    ) -> ExplorerConfig:
        page = self.pages[-1]
        page = page.with_viz(
            viz=viz,
            type=type,
            params=params,
            name=name,
        )
        # return replace(
        #     self, pages = self.pages[:-1] + [page]
        # )
        self.pages[-1] = page
        return self


@dataclass(frozen=True)
class Explorer:
    graphs: dict[str, str] = field(default_factory=dict)
    config: ExplorerConfig | None = None
    root: str = "./__local__/apps"

    @classmethod
    def load(cls):
        raise ValueError(cls)

    def reset(self):
        if self.config is not None:
            self.config.write(root=self.root)
        return self

    def run(self):
        raise ValueError()

    # TODO: rename page
    # rename graph to viz
    # rename viz to Vis? component?
    # obj -> data and viz to obj?

    def with_graph(
        self,
        graph: Graph,
        t: Type[TExplorer],
    ) -> TExplorer:
        self = replace(
            self,
            graphs={
                **self.graphs,
                **{graph.name: graph.name},
            },
        )
        return assert_is(self, t)

    def with_graphs(
        self,
        t: Type[TExplorer],
        **graphs: Graph,
    ) -> TExplorer:
        return assert_is(
            replace(
                self,
                graphs={
                    **self.graphs,
                    **{
                        k: g.name for k, g in graphs.items()
                    },
                },
            ),
            t,
        )

    def with_config(self, config: ExplorerConfig):
        return replace(self, config=config)

    def with_root(self, root: str):
        return replace(self, root=root)

    def with_page(
        self,
        page: str | ExplorerPage,
        t: Type[TExplorer],
        type: str | None = None,
        graph: str | bool | None = None,
    ) -> TExplorer:
        assert self.config is not None, self
        if isinstance(graph, bool) and graph:
            graph = (
                page if isinstance(page, str) else page.name
            )
        if graph is not None:
            assert isinstance(graph, str), graph
        config = self.config.with_page(
            page,
            type=type,
            graph=graph,
        )
        self = self.with_config(config)
        return assert_is(self, t)

    def with_object(
        self,
        obj: str | ExplorerObject,
        t: Type[TExplorer],
        viz: int | None = None,
        label: str | None = None,
        group: str | None = None,
        params: str | None = None,
    ) -> TExplorer:
        assert self.config is not None, self
        config = self.config.with_object(
            obj=obj,
            viz=viz,
            label=label,
            group=group,
            params=params,
        )
        self = self.with_config(config)
        return assert_is(self, t)

    def with_viz(
        self,
        t: Type[TExplorer],
        viz: int | ExplorerVis | None = None,
        type: str | None = None,
        params: str | None = None,
        name: str | None = None,
    ) -> TExplorer:
        assert self.config is not None, self
        config = self.config.with_viz(
            viz=viz,
            type=type,
            params=params,
            name=name,
        )
        self = self.with_config(config)
        return assert_is(self, t)

    def with_objects_from_template(
        self,
        name: str,
        template: str,
        objects: list[TTemplateObject],
        t: Type[TExplorer],
        params: str | None = None,  # viz params not obj
        kws: dict | None = None,
        # TODO: encode dict params automatically?
    ) -> TExplorer:
        assert self.config is not None, self
        i = len(self.config.pages[-1].vizs)
        self = self.with_viz(
            t, i, template, name=name, params=params
        )
        for obj_spec in objects:
            if len(obj_spec) == 2:
                obj_path, label = obj_spec
                kw = {} if kws is None else kws
            elif len(obj_spec) == 3:
                obj_path, label, kw = obj_spec
            else:
                raise ValueError(obj_spec)
            # TODO: encode dict params automatically?
            # pull out those like group (or move into params?)
            self = self.with_object(
                obj_path,
                t,
                viz=i,
                label=label,
                **kw,
                # group, params etc.
            )

        return assert_is(self, t)


TTemplateObject = tuple[str, str, dict] | tuple[str, str]


def prepare_app(
    name: str,
    root_fp: str,
    Project,
    suffix_root: bool = True,
):
    if suffix_root:
        root_fp = f"{root_fp}/{name}"

    app = apps.App(name=name, root=root_fp, pages={})
    return legacy.apps.app(
        [
            app.list_pages(
                # TODO: pass the graph funcs to check for new pages?
            )
        ]
        + list(
            app.get_pages(
                daily_cls=ExploreDailyFromRows,
                Project=Project,
                # ...
            ).values()
        ),
        init_elements=app.init_page,
    )


TPage = TypeVar("TPage", bound=ExplorerPage)
TExplorer = TypeVar("TExplorer", bound=Explorer)

__all__ = [
    "ExplorerObject",
    "ExplorerVis",
    "ExplorerPage",
    "ExplorerConfig",
    "prepare_app",
    "TPage",
    "TExplorer",
    "TTemplateObject",
]
