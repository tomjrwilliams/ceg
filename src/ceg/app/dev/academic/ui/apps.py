
from typing import Callable

import streamlit as st
from streamlit.navigation.page import StreamlitPage

def app(
    pages: dict[str, dict[
        str, Callable[
            [], 
            # StreamlitPage
            None
        ]]
    ],
    init_elements: Callable = lambda: None,
    page_config: dict = {"layout": "wide"},
):
    if len(page_config):
        st.set_page_config(**page_config)
    init_elements()
    # for _, ps in pages.items():
    #     for name, p in ps.items():
    #         p.__name__ = name
    return st.navigation(
        {
            section: list(ps.values())  # type: ignore
            for section, ps in pages.items()
        }
    )
