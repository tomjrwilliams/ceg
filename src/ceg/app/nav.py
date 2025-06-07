from typing import Callable, Any, NamedTuple
from frozendict import frozendict
import streamlit as st

from streamlit.navigation.page import StreamlitPage

Shared = frozendict[str, Any]

class Page_Kw(NamedTuple):
    shared: Shared

class Page_Interface:
    shared: Shared
    def run(self) -> StreamlitPage: ...

class Page(Page_Kw, Page_Interface):
    pass

def page(pages: frozendict[str, tuple[Page, ...]]):
    return st.navigation({
        section: [p.run for p in ps] # type: ignore
        for section, ps in pages.items()
    })

