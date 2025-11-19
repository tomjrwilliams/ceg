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

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ..variables import *

@dataclass(frozen=True)
class Page:
    pass

@dataclass(frozen=True)
class Component:
    key: str
    
    def child_key(self, key: str) -> str:
        return f"{self.key}_{key}"
    
    def render(self) -> Component: ...

    def page(self) -> Callable[[], None]:
        def f():
            self.render()
            return
        return f
    