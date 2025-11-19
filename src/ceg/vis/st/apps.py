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
import json

from dataclasses import replace, field

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from .components import *

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

@dataclass(frozen=True)
class App:
    name: str
    root: str

    pages: dict[str, Any] = field(
        default_factory=dict
    )

    def list_pages(self):
        def f():
            for dp in Path(self.root).iterdir():
                if dp.is_file():
                    continue
                fp_config = dp / "config.json"
                if not fp_config.exists():
                    continue
                if dp.name in self.pages:
                    st.page_link(self.pages[dp.name])
        return st.Page(
            f, title="pages", default=True,
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

        with st.sidebar:
            st.text_input(
                "name:", key=f"{self.name}.sidenav.name"
            )
            st.text_input(
                "type:", key=f"{self.name}.sidenav.type"
            )

            hidden_link(self.get_page_name(), "RENAME")
            hidden_link(self.get_page_name(), "NEW")
            
            st.button("RENAME", on_click=click_rename)
            st.button("NEW", on_click=click_new)

    def get_pages(
        self,
        obj_to_daily
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
                f = ExploreDailyFromRows(
                    name,
                    str(dp),
                    lambda r: obj_to_daily(**r),
                    lambda r: (
                        r["label"] 
                        if r["label"] is not None
                        else r["obj"]
                    ),
                    schema={
                        "graph": pl.Int32,
                        "obj": pl.String,
                        "label": pl.String,
                        "group": pl.String,
                        "params": pl.String,
                    },
                ).page()
                page = st.Page(
                    f, title=name, url_path = name
                )
                pages[name] = page
                self.pages[name] = page
            else:
                raise ValueError(config)
            
        return pages