import os
import sys
sys.path.append("./src")

import json
import pathlib

from typing import Any, cast, Callable
from functools import wraps

from frozendict import frozendict

import streamlit as st

os.environ["STREAMLIT"] = "true"
os.environ["TIMEZONE_OFFSET"] = str(st.context.timezone_offset)

import ceg
import ceg.fs as fs
import ceg.data as data
import ceg.app as app

import ceg.app.examples as examples

shared: app.Shared = cast(app.Shared, frozendict())

pages: frozendict[
    str, tuple[app.nav.Page, ...]
] = frozendict() # type: ignore

pages = (
    pages.set("bars", (
        examples.bars.lines("ES"),
        examples.bars.lines("CL"),
    )).set("vol", (
        examples.vol.lines("ES"),
        examples.vol.lines("CL"),
    ))
)

app.nav.page(pages, page_config=dict(
    page_title="ceg",
    layout="wide"
)).run()