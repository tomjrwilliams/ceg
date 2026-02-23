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

from .... import apis
st = apis.st.library()

from dataclasses import replace

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

from ..variables import *

# @dataclass(frozen=True)
# class Page:
#     pass

@dataclass(frozen=True)
class Component:
    key: str
    
    def child_key(self, key: str) -> str:
        return f"{self.key}_{key}"
    
    def pre_render(self, Project = None) -> None: ...

    def render(self, Project = None) -> Component: ...

    def post_render(self, Project = None) -> None: ...

    def page(self, Project) -> Callable[[], None]:
        def f():
            self.pre_render(Project)
            self.render(Project)
            self.post_render(Project)
            return
        return f
    
__all__ = ["Component"]