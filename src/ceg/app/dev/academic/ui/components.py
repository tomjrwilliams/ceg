from typing import Callable

import numpy as np
import polars as pl
import streamlit as st

from .. import core

def session_state_text_input(
    label: str,
    variable: str,
    state_key: str | None = None,
    default: str | None = None,
):
    if state_key is None:
        state_key = variable + "_"
    st.session_state[variable] = st.text_input(
        label, 
        key=state_key,
        value=default
    )

import os
from functools import lru_cache

def cache():
    # TODO: move to shared utils lib
    if os.environ.get("STREAMLIT") == "true":
        def st_decorator(f):
            import streamlit as st
            return st.cache_data(f)
        return st_decorator
    else:
        def lru_decorator(f):
            return lru_cache(maxsize=512)(f)
        return lru_decorator

@cache()
def load_results(fp: str, **kwargs):
    return core.results.Results.load(fp, **kwargs)

def reset_cache_button():
    st.button(
        "reset",
        on_click=lambda: st.cache_data.clear()
    )

# st.table(df)
# st.dataframe(df)
# st.data_editor(df)

def single_table(
    name: str,
    to_df: Callable,
    trial="000",
    f_data: Callable[
        [np.ndarray], np.ndarray
    ] | None = None,
    f_df: Callable[
        [pl.DataFrame], pl.DataFrame
    ] | None = None,
    height: int | None =300,
):
    def res():
        sim = st.session_state["SIM"]
        if sim is None or not len(sim):
            return
        fp = f"./__local__/{sim}/trial-{trial}/results.npz"
        data = load_results(fp, **{name: True})
        if f_data is not None:
            data=f_data(data)
        df = to_df(data)
        if f_df is not None:
            df = f_df(df)
        if height is not None:
            with st.container(height=height):
                st.table(df)
        else:
            st.table(df, height=height)
    return res
