from typing import Callable, Any, NamedTuple
from frozendict import frozendict
import streamlit as st

from streamlit.navigation.page import StreamlitPage

Shared = frozendict[str, Any]

class Page_Kw(NamedTuple):
    name: str
    shared: Shared

class Page_Interface:
    name: str
    shared: Shared
    def run(self) -> StreamlitPage: ...

    def named_run(self):
        run = lambda: self.run()
        run.__name__ = self.name
        return run

class Page(Page_Kw, Page_Interface):
    pass

def page(
    pages: frozendict[str, tuple[Page, ...]],
    page_config: dict = {
        "layout": "wide"
    }
):
    if len(page_config):
        st.set_page_config(**page_config)
    return st.navigation({
        section: [p.named_run() for p in ps] # type: ignore
        for section, ps in pages.items()
    })

