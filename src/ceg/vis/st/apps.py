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

from ... import apis
st = apis.st.library()

import plotly.express as px
import json

from dataclasses import replace, field

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from .components import *

logger, log = apis.logs.logger(
    __file__,
    streams=dict(formatter=True),
)

# TODO: this is really an objs.vis file (move)

def hidden_link(path: str | None, id:str):
    if path is None:
        return
    root = st.context.url
    if root is None:
        return
    root = root.split("/")[0]
    st.write(
        f"<a href='{root}/{path}' id='{id}'></a>", unsafe_allow_html=True
    )

@apis.st.cache()
def load_configs(root: str):
    res = []
    for dp in Path(root).iterdir():
        if dp.is_file():
            continue
        fp_config = dp / "config.json"
        if not fp_config.exists():
            continue
        res.append(dp)
    return res

@dataclass(frozen=True)
class App:
    name: str
    root: str

    pages: dict[str, Any] = field(
        default_factory=dict
    )

    def list_pages(self):
        # log("INFO")( build_page_list=self.root)
        def f():
            # log("INFO")( loading_configs=self.root)
            dps = load_configs(self.root)
            for dp in dps:
                if dp.name in self.pages:
                    st.page_link(self.pages[dp.name])
            # log("INFO")( loaded_configs=self.root)
        return st.Page(
            f,
            title="home",
            default=True,
        )

    def get_page_name(self):
        return st.session_state.get(
            f"{self.name}.sidenav.name"
        )
    def get_page_type(self):
        return st.session_state.get(
            f"{self.name}.sidenav.type"
        )
        
    def init_page(self):
        def click_rename():
            url = st.context.url
            if url is None:
                return
            url_split = url.split("/")
            if len(url_split) > 1:
                path = url_split[-1]
                name = self.get_page_name()
                assert name is not None, name
                if path != name:
                    path_root = Path(self.root)
                    (path_root / path).rename(
                        path_root / name
                    )
            return
        
        def click_new():
            name = self.get_page_name()
            type = self.get_page_type()
            if name is None or type is None:
                return
            if name == "" or type == "":
                return
            dp = Path(self.root) / name
            fp_config: Path = dp / "config.json"
            fp_config.parent.mkdir(
                exist_ok=True, parents=True
            )
            with fp_config.open('w+') as f:
                json.dump({
                    "type": type,
                }, f)

        # with st.sidebar:
        #     st.text_input(
        #         "name:", key=f"{self.name}.sidenav.name"
        #     )
        #     st.text_input(
        #         "type:", key=f"{self.name}.sidenav.type"
        #     )

        #     hidden_link(self.get_page_name(), "RENAME")
        #     hidden_link(self.get_page_name(), "NEW")
            
        #     c1, c2 = st.columns(2)
        #     with c2:
        #         st.button("NEW", on_click=click_new)
        #     with c1:
        #         st.button("RENAME", on_click=click_rename)

    def get_pages(
        self,
        daily_cls: Type[ExploreDailyFromRows],
        Project,
    ):
        pages = {}
        for dp in Path(self.root).iterdir():
            if dp.is_file():
                continue
            fp_config = dp / "config.json"
            if dp.is_file():
                continue
            fp_config = dp / "config.json"
            if not fp_config.exists():
                continue
            with fp_config.open('r') as f_config:
                config = json.load(f_config)
            
            name = dp.name
            # TODO: or numeric names?

            if config["type"] == "daily":
                f = daily_cls(
                    name,
                    str(dp),
                    schema={
                        "vis": pl.Int32,
                        "obj": pl.String,
                        "label": pl.String,
                        "group": pl.String,
                        "params": pl.String,
                    },
                ).page(Project)
                page = st.Page(
                    f, title=name, url_path = name
                )
                pages[name] = page
                self.pages[name] = page
            else:
                raise ValueError(config)
    
        return pages