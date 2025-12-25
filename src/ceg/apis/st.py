from typing import Callable, TypeVar
import os
from functools import lru_cache
from contextlib import contextmanager

from dataclasses import dataclass

def _streamlit():
    import streamlit as st
    return st

T = TypeVar("T")

@dataclass()
class Mocked:
    attr: str
    
    def __call__(self, *args, **kwargs):
        return Mocked(self.attr)

    def __enter__(self):
        return 

    def __exit__(self, type, value, traceback):
        return

    def __iter__(self):
        yield from []

    def __getattr__(self, attr):
        return Mocked(attr)
    
    def __setattr__(self, k, v):
        return
    
    def __getitem__(self, i):
        return
    
    def __setitem__(self, i, v):
        return

STATE = {}

def _library(
    f: Callable[[], T],
    mock: bool = False,
) -> T:
    mock = mock or os.environ.get("MOCK_STREAMLIT") == "true"
    if not mock:
        import streamlit 
        return streamlit # type: ignore
    
    class StreamlitMock:

        def Page(self, f, *args, **kwargs):
            return f
        
        def navigation(self, pages):
            return pages
        
        def columns(self, n: int, *args, **kwargs):
            return tuple([
                Mocked("columns") for _ in range(n)
            ])
        
        def toggle(self, *args, value: bool = False, **kwargs):
            return value
        
        def data_editor(self, df, *args, **kwargs):
            return df
        
        def text_input(*args, value: str = "", **kwargs):
            return value
        
        @property
        def session_state(self):
            return STATE
        
        def __getattr__(self, attr):
            return Mocked(attr)

    return StreamlitMock() # type: ignore

def library(mock: bool = False):
    return _library(_streamlit, mock=mock)

def cache():
    # TODO: move to shared utils lib
    if os.environ.get("STREAMLIT") == "true":
        def st_decorator(f):
            st = library()
            return st.cache_data(f)
        return st_decorator
    else:
        def lru_decorator(f):
            return lru_cache(maxsize=512)(f)
        return lru_decorator
    
__all__ = ["cache", "library"]