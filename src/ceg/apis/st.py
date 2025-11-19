
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